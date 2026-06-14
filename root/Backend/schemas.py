from pydantic import BaseModel


class Message(BaseModel):
    """A single message in a conversation."""

    role: str
    content: str


class AskAgentRequest(BaseModel):
    messages: list[Message]


class AskAgentResponse(BaseModel):
    """Response from the chatbot endpoint."""

    response: str
