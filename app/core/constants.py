"""
Constants used across the codebase, e.g. API endpoints, config keys, etc.
"""


from langgraph.graph import END


class InternalRoutes:
    AVAILABLE = "/api/internal/inspection-bookings/available"
    BOOK = "/api/internal/inspection-bookings"
    MY_BOOKINGS = "/api/internal/inspection-bookings/my"

    @staticmethod
    def get_booking(booking_id: str) -> str:
        return f"/api/internal/inspection-bookings/{booking_id}"

    @staticmethod
    def cancel(booking_id: str) -> str:
        return f"/api/internal/inspection-bookings/{booking_id}/cancel"

    @staticmethod
    def property_detail(property_id: str) -> str:
        return f"/api/properties/{property_id}"

    @staticmethod
    def my_deposit(listing_id: str, user_id: str) -> str:
        return f"deposit/my/{listing_id}/{user_id}"


class ToolNames:
    """
    Constants for tool names used in the AI agent.
    """
    CHECK_AVAILABILITY = "check_availability"
    BOOK_INSPECTION = "book_inspection"
    CANCEL_INSPECTION = "cancel_inspection"
    GET_BOOKING = "get_booking"
    GET_DEPOSIT = "get_deposit"


class TableNames:
    """
    Constants for database table names.
    """
    AGENCIES = "agencies"
    AGENTS = "agents"
    INSPECTION_BOOKINGS = "inspection_bookings"
    LISTINGS = "listings"
    PROPERTIES = "properties"
    PROPERTY_ADDRESSES = "property_addresses"
    PROPERTY_TYPES = "property_types"
    V_LISTINGS = "v_listings"


class AppStateKeys:
    """
    Keys for services stored on FastAPI app.state, accessed via RunnableConfig.
    """
    CONFIGURABLE = "configurable"
    THREAD_ID = "thread_id"
    USER_ID = "user_id"
    DEPOSIT_SERVICE = "deposit_service"
    BOOKING_SERVICE = "booking_service"
    SQL_VIEW_SERVICE = "sql_view_service"
    RAG_SERVICE = "rag_service"
    SEARCH_SERVICE = "search_service"
    FORCED_INTENT = "forced_intent"
    SUBURBS = "suburbs"


class Intent:
    SEARCH = "search"
    DOCUMENT_QUERY = "document_query"
    HYBRID_SEARCH = "hybrid_search"
    BOOKING = "booking"
    CANCELLATION = "cancellation"
    BOOKING_LOOKUP = "booking_lookup"
    DEPOSIT_PAYMENT = "deposit_payment"
    SUBURB_SUMMARY = "suburb_summary"
    GENERAL = "general"
    UNKNOWN = "unknown"


class IntentConfig:
    CLASSIFIER_HISTORY_LIMIT: int = 4

    HISTORY_BY_INTENT: dict[str, int] = {
        Intent.BOOKING:         12,
        Intent.CANCELLATION:    12,
        Intent.BOOKING_LOOKUP:  6,
        Intent.SEARCH:          6,
        Intent.HYBRID_SEARCH:   6,
        Intent.DOCUMENT_QUERY:  4,
        Intent.SUBURB_SUMMARY:  4,
        Intent.GENERAL:         4,
    }

    CANCELLATION_KEYWORDS = frozenset([
        "cancel", "cancellation", "cancelled", "withdraw",
        "remove booking", "no longer want to attend", "no longer available",
        "don't want to attend", "don't want the booking", "don't want the inspection",
    ])

    BOOKING_KEYWORDS = frozenset([
        "book a viewing",
        "book an inspection",
        "book a time",
        "book a visit",
        "view the property",
        "schedule inspection",
        "schedule a viewing",
        "schedule a visit",
        "arrange inspection",
        "arrange a viewing",
        "inspect the property",
        "open for inspection",
        "open home",
    ])

    # Lookup phrases indicate the user wants to retrieve an existing booking, not create one.
    LOOKUP_KEYWORDS = frozenset([
        "my booking", "my inspection", "check my booking", "check booking",
        "booking details", "booking status", "when is my inspection",
        "what time is my", "show my booking", "my confirmation",
        "look up my booking", "find my booking",
        "see my booking", "see my inspection", "view my booking", "view my inspection",
        "booked an inspection", "booked a viewing", "i booked",
    ])

    # Suppresses the booking fast-path and detects search_then_deposit.
    SEARCH_KEYWORDS = frozenset([
        "find", "search", "show", "list", "looking for",
        "properties", "house", "apartment", "unit", "townhouse",
        "bedroom", "bathroom", "suburb", "price", "budget",
        "under", "rent for", "for rent", "to rent", "buy", "purchase",
    ])

    DEPOSIT_KEYWORDS = frozenset([
        "pay deposit",
        "paying deposit",
        "holding deposit",
        "pay the deposit",
        "deposit payment",
        "pay my deposit",
    ])

    DOCUMENT_KEYWORDS = frozenset([
        "working hours",
        "opening hours",
        "office hours",
        "business hours",
        "hours of operation",
        "trading hours",
    ])

    CONFIRMATION_KEYWORDS = frozenset([
        "yes", "confirm", "confirmed", "go ahead", "go for it", "proceed",
        "do it", "cancel it",
    ])

    BOOKING_INTENTS = frozenset([
        Intent.BOOKING,
        Intent.CANCELLATION,
        Intent.BOOKING_LOOKUP,
    ])

    DOC_INTENTS = frozenset([
        Intent.DOCUMENT_QUERY,
        Intent.HYBRID_SEARCH,
        Intent.SUBURB_SUMMARY,
    ])

    TOOL_INTENTS = frozenset([
        Intent.BOOKING,
        Intent.CANCELLATION,
        Intent.BOOKING_LOOKUP,
        Intent.DEPOSIT_PAYMENT,
    ])

    SEARCH_INTENTS = TOOL_INTENTS | frozenset([
        Intent.SEARCH,
        Intent.HYBRID_SEARCH,
    ])


class Messages:
    """User-facing fallback messages returned when the agent cannot complete a request."""
    ESCALATION = "I'm having trouble completing this — a team member will follow up shortly."
    FALLBACK = "I couldn't process that request."
    SEARCH_ERROR = "I'm having trouble finding that information right now. Please try again."
    NO_RESULTS = "No properties matched your search. Try broadening your criteria — for example, a nearby suburb or a higher price range."


class PromptLabels:
    RETRIEVED_DOCUMENTS = "[RETRIEVED DOCUMENTS]"
    PROPERTY_SEARCH_RESULTS = "[PROPERTY SEARCH RESULTS]"
    BOOKING_CONTEXT = "[BOOKING CONTEXT]"


class StateKeys:
    """
    Constants for state keys used in the AI agent.
    """
    USER_INTENT = "user_intent"
    INTENT_COMPLETED = "intent_completed"
    EARLY_RESPONSE = "early_response"
    SEARCH_CONTEXT = "search_context"
    SEARCH_RESULTS = "search_results"
    RETRIEVED_DOCS = "retrieved_docs"
    DEPOSIT_RESULT = "deposit_result"
    SUBURB_SUMMARY_RESULT = "suburb_summary_result"
    BOOKING_CONTEXT = "booking_context"
    # BOOKING_STATUS = "booking_status"
    ERROR_COUNT = "error_count"
    REQUIRES_HUMAN = "requires_human"
    LOCATION = "location"
    PHASE = "phase"


class Node:
    """
    Constants for graph node names.
    """
    INTENT = "intent"
    AGENT = "agent"
    LISTING_SEARCH = "listing_search"
    VECTOR_SEARCH = "vector_search"
    HYBRID_SEARCH = "hybrid_search"
    SQL_SEARCH = "sql_search"
    SUBURB_SUMMARY = "suburb_summary"
    TOOLS = "tools"
    CONTEXT_UPDATE = "context_update"
    SAFETY = "safety"
    END = END
