# MySQL Llama Agent

This agent receives a JSON object, queries a MySQL table, and returns a natural-language answer.

Expected table columns:

- `ID_node`
- `numPeople`
- `groupEmotion`
- `date`

`date` is stored in MySQL as a Unix timestamp in seconds. The agent converts readable date filters into Unix timestamps for SQL, then converts returned timestamps back into ISO datetimes before asking Llama to explain the result.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure Ollama is running and your local model exists:

```bash
ollama pull llama3.2
```

Create the MySQL table:

```bash
mysql -u root -p silvan < schema.sql
```

Edit the `.env` with your MySQL credentials, table name, model tag, and timezone.
For example: 

'''
  MYSQL_HOST=localhost
  MYSQL_PORT=3306
  MYSQL_USER=agent
  MYSQL_PASSWORD=agent_password
  MYSQL_DATABASE=silvan
  MYSQL_TABLE=node_observations
  OLLAMA_MODEL=llama3.2
  AGENT_TIMEZONE=Europe/Rome
'''

## JSON Contract

Your server code should call `handle_request(payload)` with a JSON object:

```python
from sylvan_agent import handle_request

response = handle_request({
    "question": "Which node had the highest number of people today?",
    "debug": False,
})

print(response["answer"])
```

Request:

```json
{
  "question": "What is the latest emotion detected by node 1?",
  "debug": false
}
```

Response:

```json
{
  "answer": "Node 1 most recently detected a happy group at 2026-06-13T10:15:00+02:00."
}
```

Set `"debug": true` to include the generated query plan, SQL, parameters, and rows.

## Local JSON Smoke Test

For local testing, the script reads one JSON object from standard input:

```bash
printf '{"question":"Show the emotion breakdown for today","debug":true}' | python codex_made/mysql_agent.py
```

## Example Questions

- `How many observations do we have for node 2 today?`
- `What is the latest emotion detected by node 1?`
- `Which node had the highest number of people yesterday?`
- `Show the emotion breakdown for last week.`
- `What is the average number of people for happy groups this month?`

## Safety

The model does not execute arbitrary SQL. Llama only produces a constrained JSON query plan, and Python builds parameterized SQL from the validated plan.
