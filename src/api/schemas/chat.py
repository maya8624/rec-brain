"""
API-layer Pydantic schemas — defines the HTTP contract.
Kept separate from tool models so API shape can evolve independently.
"""

import uuid
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1, max_length=500,
        description="User message"
    )

    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Conversation session ID. Pass the same ID across requests to maintain history.",
    )


class ChatResponse(BaseModel):
    reply: str = Field(
        ...,
        description="Agent response"
    )

    session_id: str = Field(
        ...,
        description="Session ID to use in the next request"
    )
