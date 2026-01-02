from typing import Optional

from pydantic import BaseModel


class InferenceRequest(BaseModel):
    prompt: str


class ApiResponse(BaseModel):
    response: Optional[str] = None
    error: Optional[str] = None
