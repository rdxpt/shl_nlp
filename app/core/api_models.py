from typing import List, Literal
from pydantic import BaseModel, HttpUrl

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class RecommendationItem(BaseModel):
    name: str
    url: HttpUrl
    # Expanded options to safely match SHL's actual classification values
    test_type: Literal["K", "P", "S", "C"] 

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[RecommendationItem]
    end_of_conversation: bool