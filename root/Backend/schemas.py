from pydantic import BaseModel


class AskAgentRequest(BaseModel):
    message: str
    foto: str | None = None


class AskAgentResponse(BaseModel):
    response: str
