"""
app/agents/state.py

Defines the state that flows through the LangGraph agent graph.
State persists across ALL turns of a conversation via the checkpointer.

Architecture note:
    Python AI service owns: conversation state, intent, context
    .NET backend owns:      bookings, availability, property data

    booking_context is a lightweight mirror of what .NET returned —
    the source of truth always lives in .NET. Python never stores
    booking records — it only passes data through to the LLM.

LangGraph state rules:
    messages       — Annotated with operator.add → APPENDS each turn
    everything else — plain assignment → last write wins per turn
"""
from enum import Enum
from typing import Annotated, Literal, Sequence, TypedDict
import operator
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

UserIntent = Literal[
    "search",
    "document_query",
    "hybrid_search",
    "booking",
    "cancellation",
    "booking_lookup",
    "deposit_payment",
    "general",
    "unknown"
]


class ConversationPhase(str, Enum):
    IDLE = "idle"
    SEARCH_RESULTS_SHOWN = "search_results_shown"
    BOOKING_CONFIRMED = "booking_confirmed"
    BOOKING_PENDING = "booking_pending"
    DEPOSIT_PENDING = "deposit_pending"
    DEPOSIT_CONFIRMED = "deposit_confirmed"
    CANCELLATION_PENDING = "cancellation_pending"


class PropertyContext(TypedDict, total=False):
    """
    The property currently being discussed in this conversation.
    Populated by search_listings or get_property_details tool results.
    Reset when the user shifts to a different property.
    """
    property_id: str
    address: str
    suburb: str
    price: float                # AUD — purchase price or weekly rent
    bedrooms: int
    bathrooms: int
    property_type: str          # house | apartment | townhouse | unit | villa
    agent_id: str
    agent_name: str
    agent_phone: str


class BookingContext(TypedDict, total=False):
    """
    Tracks an in-progress inspection booking across multiple turns.

    .NET owns the booking record — this is only a conversation mirror.
    All real actions (check availability, confirm, cancel) call .NET APIs
    via BookingService.

    Multi-turn flow:
        1. availability fetched → available_slots + property_id set
        2. slot selected        → user chooses from available_slots
        3. user confirms        → book_inspection called
        4. .NET books it        → confirmation_id + confirmed=True set
    """
    property_id: str
    property_address: str

    available_slots: list[str]
    selected_slot: str

    confirmed: bool
    cancelled: bool

    confirmation_id: str
    confirmed_datetime: str

    cancellation_id: str
    cancellation_reason: str


class SearchContext(TypedDict, total=False):
    """
    Current property search criteria, built up across turns.
    The user might say "in Parramatta" then "under $600k" then "3 bedrooms"
    across separate messages — this accumulates those filters.
    """
    property_id: str
    location: str
    address: str
    listing_type: str
    property_type: str
    bedrooms: int
    bathrooms: int
    max_price: float
    min_price: float
    keywords: list[str]             # ["pool", "garage", "pet friendly"]
    last_result_count: int          # how many results the last search returned
    limit: int                      # max rows to return — user-specified, capped at 10


class IntentClassification(BaseModel):
    """
    Structured output returned by the LLM intent classifier.
    Pydantic BaseModel so LangChain's with_structured_output can deserialise it.
    """
    intent: UserIntent
    early_response: str | None = None
    location: str | None = None
    address: str | None = None
    listing_type: str | None = None
    property_type: str | None = None
    bedrooms: int | None = None
    bathrooms: int | None = None
    max_price: float | None = None
    min_price: float | None = None
    limit: int | None = None


class RealEstateAgentState(TypedDict):
    """
    Complete state for one conversation thread with the real estate agent.
    Persisted across turns by LangGraph checkpointer:
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_intent: UserIntent
    property_context: PropertyContext   # property currently being discussed
    booking_context: BookingContext     # in-progress booking details
    search_context: SearchContext       # accumulated search filters
    early_response: str | None
    search_results: list[dict]
    deposit_result: dict | None
    retrieved_docs: str | None
    requires_human: bool                # True → escalate to human agent
    error_count: int                    # consecutive tool failures this session
    intent_completed: bool              # True → last intent's tool flow finished
    phase: ConversationPhase


def initial_state() -> RealEstateAgentState:
    """
    Clean state for a new conversation thread.
    Call ONCE when a new thread starts — not on every message turn.
    LangGraph rehydrates state from the checkpointer on subsequent turns.
    """
    return RealEstateAgentState(
        messages=[],
        user_intent="unknown",
        property_context=PropertyContext(),
        booking_context=BookingContext(
            available_slots=[],
            confirmed=False,
            cancelled=False,
        ),
        search_context=SearchContext(
            keywords=[],
            last_result_count=0,
            property_id=None,
        ),
        search_results=[],
        deposit_result=None,
        retrieved_docs=None,
        requires_human=False,
        error_count=0,
        intent_completed=False,
        phase=ConversationPhase.IDLE
    )
