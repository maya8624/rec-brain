"""
Pydantic models for the /chat endpoint.
These are the contract between the .NET backend and the Python AI service.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Incoming chat message from .NET backend.

    thread_id maps to the user's session in .NET.
    LangGraph uses it to isolate and rehydrate conversation state.
    """
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User's message text",
    )

    thread_id: str = Field(
        ...,
        min_length=1,
        description="Session ID from .NET — maps to LangGraph thread",
    )

    user_id: str = Field(
        ...,
        min_length=1,
        description="Authenticated user ID from .NET JWT",
    )

    property_id: str | None = Field(
        default=None,
        description="Optional property ID from .NET — lets agent personalise response with property details",
    )

    is_new_conversation: bool = Field(
        default=False,
        description="True = new thread, initialise fresh state. False = continue existing.",
    )


class PropertyListing(BaseModel):
    """A property listing returned from a search — sent to .NET for frontend rendering."""
    property_id: str
    property_url: str | None = None
    listing_id: str


class ChatResponse(BaseModel):
    """
    Response returned to .NET backend after agent processes the message.
    """
    reply: str = Field(
        description="The agent's response text to show the user"
    )

    thread_id: str = Field(
        description="Echoed back so .NET can correlate the response"
    )

    # Booking state — lets .NET frontend show appropriate UI
    # booking_confirmed: bool = Field(
    #     default=False,
    #     description="True if an inspection was successfully booked this turn",
    # )

    # booking_cancelled: bool = Field(
    #     default=False,
    #     description="True if an inspection was successfully cancelled this turn",
    # )

    # confirmation_id: str | None = Field(
    #     default=None,
    #     description="Booking reference ID if booking_confirmed is True",
    # )

    # Escalation — lets frontend show "Contact agent" button
    # requires_human: bool = Field(
    #     default=False,
    #     description="True if agent could not handle the request",
    # )

    listings: list[PropertyListing] = Field(
        default_factory=list,
        description="Structured property listings from SQL search — render as property cards",
    )

    property_id: str | None = Field(
        default=None,
        description="listing_id of the property in context, if unambiguous",
    )


class ChatErrorResponse(BaseModel):
    """Returned on errors """
    error: str
    thread_id: str
