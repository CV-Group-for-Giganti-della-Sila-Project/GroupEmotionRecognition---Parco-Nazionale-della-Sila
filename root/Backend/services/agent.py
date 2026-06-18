import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Silvan_Agent'))

from silvan_agent import handle_request


async def run_agent(message: str, foto: str | None = None) -> str:
    try:
        payload = {"question": message}
        response = handle_request(payload)
        return response.get("answer", "No answer generated.")
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc
