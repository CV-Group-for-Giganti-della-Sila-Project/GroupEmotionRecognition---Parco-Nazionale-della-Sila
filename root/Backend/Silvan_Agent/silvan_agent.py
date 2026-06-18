#!/usr/bin/env python3
"""JSON-first AI agent for a MySQL table of people/emotion observations."""

from __future__ import annotations

import json
import os
import re
import sys
import mysql.connector
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from dotenv import load_dotenv
from langchain_ollama import ChatOllama


IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DATE_TEXT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$")
NUMERIC_TEXT_RE = re.compile(r"^-?\d+(\.\d+)?$")

PEOPLE_TOTAL_RE = re.compile(
    r"\b(how many|number of|total)\b.*\b(people|persons|visitors|visited|detected)\b",
    re.IGNORECASE,
)
RECORD_COUNT_RE = re.compile(
    r"\b(records?|rows?|observations?|entries?)\b",
    re.IGNORECASE,
)
NODE_TEXT_RE = re.compile(
    r"\bnode\s*(?:id\s*)?([A-Za-z0-9_-]+)\b",
    re.IGNORECASE,
)

DEFAULT_LIMIT = 20
MAX_LIMIT = 100
MILLISECONDS_THRESHOLD = 10_000_000_000

QUERY_TYPES = {
    "raw_rows",
    "latest_rows",
    "count_rows",
    "avg_people",
    "sum_people",
    "max_people",
    "min_people",
    "emotion_breakdown",
    "node_summary",
    "daily_summary",
}

ORDER_COLUMNS = {
    "id": "`id`",
    "node_name": "`node_name`",
    "num_persone": "`num_persone`",
    "emotion": "`emotion`",
    "timestamp": "`timestamp`",
}


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: int
    user: str
    password: str
    database: str
    table: str

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        return cls(
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.getenv("MYSQL_DATABASE", "silvan"),
            table=os.getenv("MYSQL_TABLE", "node_observations"),
        )

    @property
    def quoted_table(self) -> str:
        if not IDENTIFIER_RE.match(self.table):
            raise ValueError(
                "MYSQL_TABLE must contain only letters, numbers, and underscores."
            )
        return f"`{self.table}`"


@dataclass(frozen=True)
class AgentConfig:
    model: str
    timezone_name: str
    temperature: float = 0.0
    max_limit: int = MAX_LIMIT

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            timezone_name=os.getenv("AGENT_TIMEZONE", "Europe/Rome"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0")),
            max_limit=int(os.getenv("AGENT_MAX_LIMIT", str(MAX_LIMIT))),
        )


@dataclass(frozen=True)
class AgentResult:
    answer: str
    sql: str
    params: tuple[Any, ...]
    rows: list[dict[str, Any]]
    plan: dict[str, Any]

    def to_response(self, debug: bool = False) -> dict[str, Any]:
        response = {"answer": self.answer}
        if debug:
            response.update(
                {
                    "plan": self.plan,
                    "sql": self.sql,
                    "params": self.params,
                    "rows": self.rows,
                }
            )
        return response


class MySQLInsightAgent:
    """Turns a JSON request into a safe MySQL query and natural-language answer."""

    def __init__(self, db_config: DatabaseConfig, agent_config: AgentConfig) -> None:
        self.db_config = db_config
        self.agent_config = agent_config
        self.tz = load_timezone(agent_config.timezone_name)
        self.llm = ChatOllama(
            model=agent_config.model,
            temperature=agent_config.temperature,
        )

    def ask(self, request: dict[str, Any]) -> AgentResult:
        question = get_question(request)
        plan = self._plan_query(question)
        sql, params = build_sql(plan, self.db_config, self.agent_config.max_limit)
        rows = add_readable_dates(self._run_query(sql, params), self.tz)
        answer = self._explain(question, plan, sql, params, rows)
        return AgentResult(answer=answer, sql=sql, params=params, rows=rows, plan=plan)

    def _plan_query(self, question: str) -> dict[str, Any]:
        prompt = f"""
            You are a query planner for a read-only MySQL analytics agent.

            The database has one table named {self.db_config.table} with these columns:
            - id: numeric observation identifier
            - node_name: string sensor/node identifier
            - num_persone: integer count of detected people
            - emotion: text label for the detected group emotion
            - timestamp: Unix timestamp in seconds

            The user's timezone is {self.agent_config.timezone_name}.
            Today's date in that timezone is {datetime.now(self.tz).date().isoformat()}.

            Return exactly one JSON object and no Markdown. Use this schema:
            {{
            "query_type": "raw_rows | latest_rows | count_rows | avg_people | sum_people | max_people | min_people | emotion_breakdown | node_summary | daily_summary",
            "filters": {{
                "id": integer or null,
                "node_name": string or null,
                "emotion": string or null,
                "timestamp_from": "YYYY-MM-DD or YYYY-MM-DD HH:MM:SS" or null,
                "timestamp_to": "YYYY-MM-DD or YYYY-MM-DD HH:MM:SS" or null
            }},
            "order_by": "id | timestamp | num_persone | node_name | emotion | null",
            "order_dir": "ASC | DESC | null",
            "limit": integer or null,
            "reason": "short explanation of the data needed"
            }}

            Planning rules:
            - Pick count_rows ONLY for questions asking how many records, rows, observations, or entries exist.
            - Pick sum_people for questions asking how many people, visitors, persons, or detected people were present/visited.
            - Example: "How many people visited node node-2?" means SUM(num_persone) WHERE node_name = "node-2", not COUNT(*).
            - Pick emotion_breakdown for questions about distribution of emotions.
            - Pick node_summary for comparisons grouped by node.
            - Pick daily_summary for trends grouped by day.
            - Pick latest_rows when the user asks for the most recent/latest observations.
            - Pick raw_rows for filtered lists of observations.
            - Convert relative dates such as today, yesterday, this week, or last month into timestamp_from/timestamp_to.
            - Use timestamp_to as an exclusive upper bound.

            User question: {question}
            """
        response = self.llm.invoke(prompt)
        content = getattr(response, "content", str(response))

        # Small local models sometimes ignore "JSON only" and return SQL.
        # Do not execute model-written SQL; build a safe internal plan instead.
        try:
            raw_plan = extract_json_object(content)
        except ValueError:
            raw_plan = plan_from_question(question)

        plan = normalize_plan(raw_plan, self.tz)
        return apply_question_overrides(question, plan)

    def _run_query(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        connection = mysql.connector.connect(  # type: ignore[union-attr]
            host=self.db_config.host,
            port=self.db_config.port,
            user=self.db_config.user,
            password=self.db_config.password,
            database=self.db_config.database,
        )

        try:
            cursor = connection.cursor(dictionary=True)
            try:
                cursor.execute(sql, params)
                return list(cursor.fetchall())
            finally:
                cursor.close()
        finally:
            connection.close()

    def _explain(
        self,
        question: str,
        plan: dict[str, Any],
        sql: str,
        params: tuple[Any, ...],
        rows: list[dict[str, Any]],
    ) -> str:
        prompt = f"""
            You are a concise data analyst.

            Answer the user's question in natural language using only the SQL result rows.
            The database stores timestamp as a Unix timestamp, but rows below expose timestamp as a readable ISO datetime.
            If there are no rows, say that no matching records were found.
            Do not invent values that are not present in the rows.

            User question:
            {question}

            Query plan:
            {json.dumps(plan, indent=2)}

            SQL executed:
            {sql}

            SQL parameters:
            {json.dumps(params)}

            Rows:
            {json.dumps(rows, indent=2)}
            """
        response = self.llm.invoke(prompt)
        return getattr(response, "content", str(response)).strip()


def handle_request(
    payload: dict[str, Any],
    db_config: DatabaseConfig | None = None,
    agent_config: AgentConfig | None = None,
) -> dict[str, Any]:
    """Entry point for API/server code that receives a JSON object."""
    load_dotenv()
    agent = MySQLInsightAgent(
        db_config or DatabaseConfig.from_env(),
        agent_config or AgentConfig.from_env(),
    )
    result = agent.ask(payload)
    return result.to_response(debug=bool(payload.get("debug")))



def get_question(payload: dict[str, Any]) -> str:

    question = payload.get("question", payload.get("prompt"))
    if not isinstance(question, str) or not question.strip():
        raise ValueError("The JSON object must contain a non-empty `question` string.")
    return question.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"The model did not return JSON: {text}") from None
        parsed = json.loads(cleaned[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("The model returned JSON, but it was not an object.")
    return parsed


def normalize_plan(plan: dict[str, Any], tz: timezone) -> dict[str, Any]:
    query_type = value_in_set(plan.get("query_type"), QUERY_TYPES, "raw_rows")
    filters = plan.get("filters") if isinstance(plan.get("filters"), dict) else {}

    return {
        "query_type": query_type,
        "filters": {
            "id": as_int(filters.get("id")),
            "node_name": as_text(filters.get("node_name")),
            "emotion": as_text(filters.get("emotion")),
            "timestamp_from": datetime_filter_to_unix(filters.get("timestamp_from"), tz),
            "timestamp_to": datetime_filter_to_unix(filters.get("timestamp_to"), tz),
        },
        "order_by": value_in_set(plan.get("order_by"), set(ORDER_COLUMNS), None),
        "order_dir": value_in_set(upper_text(plan.get("order_dir")), {"ASC", "DESC"}, None),
        "limit": as_int(plan.get("limit")),
        "reason": as_text(plan.get("reason")),
    }



def plan_from_question(question: str) -> dict[str, Any]:
    """Fallback planner used when the LLM does not return valid JSON."""
    query_type = "raw_rows"

    if PEOPLE_TOTAL_RE.search(question) and not RECORD_COUNT_RE.search(question):
        query_type = "sum_people"
    elif RECORD_COUNT_RE.search(question):
        query_type = "count_rows"
    elif re.search(r"\b(avg|average|mean)\b.*\b(people|persons|visitors)\b", question, re.IGNORECASE):
        query_type = "avg_people"
    elif re.search(r"\b(max|maximum|highest|most)\b.*\b(people|persons|visitors)\b", question, re.IGNORECASE):
        query_type = "max_people"
    elif re.search(r"\b(min|minimum|lowest|least)\b.*\b(people|persons|visitors)\b", question, re.IGNORECASE):
        query_type = "min_people"
    elif re.search(r"\b(latest|most recent|recent)\b", question, re.IGNORECASE):
        query_type = "latest_rows"

    node_match = NODE_TEXT_RE.search(question)

    return {
        "query_type": query_type,
        "filters": {
            "id": None,
            "node_name": node_match.group(1) if node_match else None,
            "emotion": None,
            "timestamp_from": None,
            "timestamp_to": None,
        },
        "order_by": None,
        "order_dir": None,
        "limit": None,
        "reason": "Fallback plan generated from the user's question because the model did not return JSON.",
    }


def apply_question_overrides(question: str, plan: dict[str, Any]) -> dict[str, Any]:
    """Correct common LLM planning mistakes with deterministic rules."""
    if PEOPLE_TOTAL_RE.search(question) and not RECORD_COUNT_RE.search(question):
        plan["query_type"] = "sum_people"

        node_match = NODE_TEXT_RE.search(question)
        if node_match and plan["filters"].get("node_name") is None:
            plan["filters"]["node_name"] = node_match.group(1)

        plan["reason"] = (
            "The user asked for a total number of people, so use SUM(num_persone), "
            "not COUNT(*), which counts observations."
        )

    return plan


def build_sql(
    plan: dict[str, Any],
    db_config: DatabaseConfig,
    max_limit: int = MAX_LIMIT,
) -> tuple[str, tuple[Any, ...]]:
    table = db_config.quoted_table
    query_type = plan["query_type"]
    where_sql, params = build_where_clause(plan["filters"])
    limit = clamp_limit(plan.get("limit"), max_limit)

    if query_type == "count_rows":
        return f"SELECT COUNT(*) AS observation_count FROM {table}{where_sql}", tuple(params)

    if query_type == "avg_people":
        return (
            f"SELECT ROUND(AVG(`num_persone`), 2) AS avg_people FROM {table}{where_sql}",
            tuple(params),
        )

    if query_type == "sum_people":
        return (
            f"SELECT COALESCE(SUM(`num_persone`), 0) AS total_people FROM {table}{where_sql}",
            tuple(params),
        )

    if query_type == "emotion_breakdown":
        sql = (
            "SELECT `emotion`, COUNT(*) AS observation_count, "
            "ROUND(AVG(`num_persone`), 2) AS avg_people, "
            "SUM(`num_persone`) AS total_people "
            f"FROM {table}{where_sql} "
            "GROUP BY `emotion` "
            "ORDER BY observation_count DESC, `emotion` ASC "
            "LIMIT %s"
        )
        return sql, tuple(params + [limit])

    if query_type == "node_summary":
        sql = (
            "SELECT `node_name`, COUNT(*) AS observation_count, "
            "ROUND(AVG(`num_persone`), 2) AS avg_people, "
            "MAX(`num_persone`) AS max_people, "
            "MIN(`num_persone`) AS min_people "
            f"FROM {table}{where_sql} "
            "GROUP BY `node_name` "
            "ORDER BY `node_name` ASC "
            "LIMIT %s"
        )
        return sql, tuple(params + [limit])

    if query_type == "daily_summary":
        sql = (
            "SELECT DATE(FROM_UNIXTIME(`timestamp`)) AS day, "
            "COUNT(*) AS observation_count, "
            "ROUND(AVG(`num_persone`), 2) AS avg_people, "
            "SUM(`num_persone`) AS total_people "
            f"FROM {table}{where_sql} "
            "GROUP BY DATE(FROM_UNIXTIME(`timestamp`)) "
            "ORDER BY day ASC "
            "LIMIT %s"
        )
        return sql, tuple(params + [limit])

    if query_type in {"max_people", "min_people"}:
        direction = "DESC" if query_type == "max_people" else "ASC"
        sql = (
            "SELECT `id`, `node_name`, `num_persone`, `emotion`, `timestamp` "
            f"FROM {table}{where_sql} "
            f"ORDER BY `num_persone` {direction}, `timestamp` DESC "
            "LIMIT %s"
        )
        return sql, tuple(params + [limit])

    order_clause = " ORDER BY `timestamp` DESC"
    if query_type == "raw_rows":
        order_clause = build_order_clause(plan.get("order_by"), plan.get("order_dir"))

    sql = (
        "SELECT `id`, `node_name`, `num_persone`, `emotion`, `timestamp` "
        f"FROM {table}{where_sql}{order_clause} LIMIT %s"
    )
    return sql, tuple(params + [limit])


def build_where_clause(filters: dict[str, Any]) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []

    for key, column, operator in (
        ("id", "`id`", "="),
        ("node_name", "`node_name`", "="),
        ("emotion", "`emotion`", "="),
        ("timestamp_from", "`timestamp`", ">="),
        ("timestamp_to", "`timestamp`", "<"),
    ):
        value = filters.get(key)
        if value is not None:
            clauses.append(f"{column} {operator} %s")
            params.append(value)

    return (" WHERE " + " AND ".join(clauses), params) if clauses else ("", params)


def build_order_clause(order_by: str | None, order_dir: str | None) -> str:
    column = ORDER_COLUMNS.get(order_by or "", "`timestamp`")
    direction = order_dir if order_dir in {"ASC", "DESC"} else "DESC"
    return f" ORDER BY {column} {direction}"


def unix_timestamp_to_datetime(value: Any, tz: timezone | ZoneInfo) -> str:
    """Convert Unix seconds or milliseconds into a readable datetime."""
    return datetime.fromtimestamp(unix_seconds(value), tz).isoformat(timespec="seconds")


def datetime_filter_to_unix(value: Any, tz: timezone | ZoneInfo) -> int | None:
    if is_blank(value):
        return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=tz)
        return int(dt.timestamp())
    if isinstance(value, date):
        return int(datetime.combine(value, time.min, tz).timestamp())

    text = str(value).strip()
    if NUMERIC_TEXT_RE.match(text):
        return unix_seconds(text)
    if not DATE_TEXT_RE.match(text):
        return None

    dt = datetime.fromisoformat(text.replace(" ", "T"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return int(dt.timestamp())


def add_readable_dates(
    rows: list[dict[str, Any]],
    tz: timezone | ZoneInfo,
) -> list[dict[str, Any]]:
    converted_rows = []
    for row in rows:
        converted = dict(row)
        if converted.get("timestamp") is not None:
            converted["timestamp_unix"] = unix_seconds(converted["timestamp"])
            converted["timestamp"] = unix_timestamp_to_datetime(converted["timestamp"], tz)
            
        # Convert Decimals so JSON can serialize them
        for key, value in converted.items():
            if isinstance(value, Decimal):
                # Convert to int if it's a whole number, otherwise float
                converted[key] = int(value) if value % 1 == 0 else float(value)
                
        converted_rows.append(converted)
    return converted_rows


# IF POSSIBLE REMOVE THIS FUNCTIONS
def unix_seconds(value: Any) -> int:
    timestamp = int(float(value))
    if abs(timestamp) >= MILLISECONDS_THRESHOLD:
        return timestamp // 1000
    return timestamp


def clamp_limit(value: Any, max_limit: int) -> int:
    limit = as_int(value) or DEFAULT_LIMIT
    return max(1, min(limit, max_limit))


def as_int(value: Any) -> int | None:
    if is_blank(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_text(value: Any) -> str | None:
    if is_blank(value):
        return None
    return str(value).strip()


def upper_text(value: Any) -> str | None:
    text = as_text(value)
    return text.upper() if text else None


def value_in_set(value: Any, allowed: set[str], fallback: Any) -> Any:
    text = as_text(value)
    return text if text in allowed else fallback


def is_blank(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip().lower() in {"", "null", "none"}
    )


def load_timezone(timezone_name: str) -> timezone | ZoneInfo:
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        return timezone.utc



def main() -> int:
    payload = json.load(sys.stdin)
    response = handle_request(payload)
    print(json.dumps(response, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
