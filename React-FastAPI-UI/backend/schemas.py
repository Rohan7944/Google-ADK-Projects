from pydantic import BaseModel

class ChatRequest(BaseModel):
    user_message: str
    agent_id: str
    project_id: str
    location: str


class ChatResponse(BaseModel):
    response_text: str