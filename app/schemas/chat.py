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
    A new thread_id = fresh conversation.
    Same thread_id across messages = continuing conversation.
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


class Listing(BaseModel):
    """A property listing returned from a search — sent to .NET for frontend rendering."""
    listing_id: str
    address: str
    suburb: str
    state: str
    postcode: str
    price: float
    bedrooms: int
    bathrooms: int
    car_spaces: int = 0
    property_type: str
    listing_type: str   # Sale | Rent
    listing_status: str
    agent_name: str = ""
    agent_phone: str = ""
    agency_name: str = ""


class SourceDocument(BaseModel):
    """A source document cited in a RAG response."""
    document: str
    doc_type: str = ""
    page: str = ""
    relevance_score: float = 0.0


class ChatResponse(BaseModel):
    """
    Response returned to .NET backend after agent processes the message.

    The .NET backend uses the metadata fields to drive frontend behaviour:
        tools_used        → show "Searched listings" / "Checked availability" indicators
        booking_initiated → show booking confirmation UI
        requires_human    → show "Contact agent" CTA
        intent            → analytics / logging
    """
    reply: str = Field(
        description="The agent's response text to show the user"
    )

    thread_id: str = Field(
        description="Echoed back so .NET can correlate the response"
    )

    # What happened this turn
    tools_used: list[str] = Field(
        default_factory=list,
        description="Names of tools called this turn eg ['search_listings']",
    )

    intent: str = Field(
        default="unknown",
        description="Detected intent: search | document_query | booking | cancellation | general",
    )

    # Booking state — lets .NET frontend show appropriate UI
    booking_confirmed: bool = Field(
        default=False,
        description="True if an inspection was successfully booked this turn",
    )

    booking_cancelled: bool = Field(
        default=False,
        description="True if an inspection was successfully cancelled this turn",
    )

    confirmation_id: str | None = Field(
        default=None,
        description="Booking reference ID if booking_confirmed is True",
    )

    # Escalation — lets frontend show "Contact agent" button
    requires_human: bool = Field(
        default=False,
        description="True if agent could not handle the request",
    )

    # Property listings — lets frontend render a property list
    listings: list[Listing] = Field(
        default_factory=list,
        description="Structured property listings from SQL search — render as property cards",
    )

    # Property ID — populated when a single property is in context (eg. direct address lookup)
    # lets .NET show the "Book Inspection" button for that specific property
    property_id: str | None = Field(
        default=None,
        description="listing_id of the property in context, if unambiguous",
    )

    # RAG sources — lets frontend show "Sources" section
    sources: list[SourceDocument] = Field(
        default_factory=list,
        description="Source documents cited if search_documents was called",
    )


class ChatErrorResponse(BaseModel):
    """Returned on errors """
    error: str
    thread_id: str
