from pydantic import BaseModel


class AskAgentRequest(BaseModel):
    message: str
    foto: str | None = None


class AskAgentResponse(BaseModel):
    response: str


class AnalyzePhotoRequest(BaseModel):
    image_base64: str


class AnalyzePhotoResponse(BaseModel):
    emotion: str
