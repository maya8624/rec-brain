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
from typing import Annotated, Literal, Sequence, TypedDict
import operator
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

# ------------------------------------
# Intent literals)
# Explicit Literal keeps the LLM honest — only these values are valid
# ------------------------------------
UserIntent = Literal[
    "search",           # user wants to find properties
    "document_query",   # user asking about leases, contracts, strata
    "hybrid_search",    # user wants property listings + document context together
    "booking",          # user wants to inspect a property
    "cancellation",     # user wants to cancel an existing inspection
    "booking_lookup",   # user wants to check details of an existing booking
    # user wants to search first, then book — run search, then auto-proceed to check_availability
    "search_then_book",
    "general",          # general question about the agency / process
    "unknown",          # intent not yet determined
]


# ------------------------------------
#  Nested context types
# ------------------------------------

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
        1. intent detected     → property_id set
        2. availability fetched → available_slots populated (.NET API call)
        3. slot selected       → selected_slot set
        4. contact collected   → contact_* fields filled turn by turn
        5. user confirms       → awaiting_confirmation=True
        6. .NET books it       → confirmation_id set, confirmed=True
    """
    # Property being inspected
    property_id: str
    property_address: str

    # Slots returned by .NET availability API
    available_slots: list[str]      # ["2025-06-14 10:00", "2025-06-14 14:00"]
    selected_slot: str              # slot the user chose from available_slots

    # Contact details — collected across turns, sent to .NET on confirmation
    contact_name: str
    contact_email: str
    contact_phone: str

    # Set by .NET after successful booking
    confirmation_id: str            # .NET booking reference number
    confirmed_datetime: str         # confirmed slot from .NET response

    # Set when cancelling an existing booking
    cancellation_id: str            # existing booking ID to cancel
    cancellation_reason: str


class BookingStatus(TypedDict, total=False):
    """
    Separate from BookingContext — tracks WHERE we are in the booking flow.
    Using explicit bool fields with required=True (no total=False) so they
    are always present and never cause KeyError.
    """
    awaiting_confirmation: bool     # all details collected, needs user yes/no
    confirmed: bool                 # booking completed with .NET
    cancelled: bool                 # booking was cancelled with .NET


class SearchContext(TypedDict, total=False):
    """
    Current property search criteria, built up across turns.
    The user might say "in Parramatta" then "under $600k" then "3 bedrooms"
    across separate messages — this accumulates those filters.
    """
    property_id: str                # specific property ID passed from .NET (eg. from property page)
    location: str                   # suburb or area name
    # street address (e.g. "177 Castlereagh St")
    address: str
    listing_type: str               # "Sale" or "Rent"
    property_type: str              # House | Apartment | Townhouse | Villa | Studio
    bedrooms: int
    bathrooms: int
    max_price: float                # AUD
    min_price: float                # AUD
    keywords: list[str]             # ["pool", "garage", "pet friendly"]
    last_result_count: int          # how many results the last search returned
    limit: int                      # max rows to return — user-specified, capped at 10


# ------------------------------------
#  LLM-based intent classification output
# ------------------------------------
class IntentClassification(BaseModel):
    """
    Structured output returned by the LLM intent classifier.
    Pydantic BaseModel so LangChain's with_structured_output can deserialise it.
    """
    intent: UserIntent
    early_response: str | None = None
    # Search entities — null means not mentioned, do not guess
    location: str | None = None
    # street address (e.g. "177 Castlereagh St")
    address: str | None = None
    listing_type: str | None = None     # "Sale" or "Rent"
    # House | Apartment | Townhouse | Villa | Studio
    property_type: str | None = None
    bedrooms: int | None = None
    bathrooms: int | None = None
    max_price: float | None = None
    min_price: float | None = None
    # explicit count requested by the user (e.g. "show me 3")
    limit: int | None = None


# ── Main agent state ──────────────────────────────────────────────────────────

class RealEstateAgentState(TypedDict):
    """
    Complete state for one conversation thread with the real estate agent.
    Persisted across turns by LangGraph checkpointer:
    """

    # ── Conversation history ──────────────────────────────────────────────────
    # operator.add = APPEND semantics — never overwrite the full list
    # In nodes, always return: {"messages": [new_message]}
    # Never return: {"messages": all_messages}
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # ── Intent ────────────────────────────────────────────────────────────────
    # Detected from the latest user message
    # Router uses this to short-circuit to the right tool fast
    user_intent: UserIntent

    # ── Structured context ────────────────────────────────────────────────────
    # Built incrementally across turns — nodes merge into these dicts
    property_context: PropertyContext   # property currently being discussed
    booking_context: BookingContext     # in-progress booking details
    booking_status: BookingStatus       # where we are in the booking flow
    search_context: SearchContext       # accumulated search filters

    # set by intent_node for compound intents, bypasses LLM
    early_response: str | None

    # ── Search results ────────────────────────────────────────────────────────
    # Slim property rows from the last SQL search — returned in ChatResponse
    # for the frontend to render as property cards. Reset each search turn.
    search_results: list[dict]

    # Current-turn search/RAG content — plain assignment, so always holds only
    # the latest turn's data. agent_node injects this directly into the LLM
    # prompt (never appended to messages), then clears it to None.
    # Prevents old search results from accumulating in conversation history.
    retrieved_docs: str | None

    # ── Flow control ──────────────────────────────────────────────────────────
    requires_human: bool                # True → escalate to human agent
    error_count: int                    # consecutive tool failures this session
    intent_completed: bool              # True → last intent's tool flow finished
    last_intent: UserIntent | None      # intent from the just-completed flow


# ------------------------------------
# Default state factory
# ------------------------------------

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
        booking_context=BookingContext(available_slots=[]),
        booking_status=BookingStatus(
            awaiting_confirmation=False,
            confirmed=False,
            cancelled=False,
        ),
        search_context=SearchContext(
            keywords=[],
            last_result_count=0,
            property_id=None,
        ),
        search_results=[],
        retrieved_docs=None,
        requires_human=False,
        error_count=0,
        intent_completed=False,
        last_intent=None,
    )
