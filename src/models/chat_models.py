from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Incoming request from the .NET backend to the Python AI service.
    """

    message: str = Field(
        ...,
        min_length=1,
        description="The user's chat message."
    )

    session_id: str | None = Field(
        default=None,
        description="Conversation/session identifier used to group related chat messages.",
    )

    user_id: str | None = Field(
        default=None,
        description="Optional user identifier from the backend system.",
    )


class ChatResponse(BaseModel):
    """
    Final response returned by the Python AI service.
    """

    answer: str = Field(
        ...,
        description="Final AI-generated response to return to the user."
    )

    session_id: str | None = Field(
        default=None,
        description="Conversation/session identifier echoed back to the caller.",
    )

    route: str | None = Field(
        default=None,
        description="The orchestration route selected for this request.",
    )

    success: bool = Field(
        default=True,
        description="Indicates whether the request was processed successfully.",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional non-sensitive metadata for debugging or observability.",
    )
