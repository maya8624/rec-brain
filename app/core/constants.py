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

# class InspectionBookingEndpoints:
#     """
#     Constants for the .NET backend API.
#     """
#     AVAILABLE = "/api/internal/inspection-bookings/available"
#     BOOK = "/api/internal/inspection-bookings"
#     CANCEL = "/api/internal/inspection-bookings/{id}/cancel"


class ToolNames:
    """
    Constants for tool names used in the AI agent.
    """
    CHECK_AVAILABILITY = "check_availability"
    BOOK_INSPECTION = "book_inspection"
    CANCEL_INSPECTION = "cancel_inspection"
    GET_BOOKING = "get_booking"


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


class StateKeys:
    """
    Constants for state keys used in the AI agent.
    """
    USER_INTENT = "user_intent"
    LAST_INTENT = "last_intent"
    INTENT_COMPLETED = "intent_completed"
    EARLY_RESPONSE = "early_response"
    SEARCH_CONTEXT = "search_context"
    SEARCH_RESULTS = "search_results"
    RETRIEVED_DOCS = "retrieved_docs"
    BOOKING_CONTEXT = "booking_context"
    BOOKING_STATUS = "booking_status"
    ERROR_COUNT = "error_count"
    REQUIRES_HUMAN = "requires_human"


class AppStateKeys:
    """
    Keys for services stored on FastAPI app.state, accessed via RunnableConfig.
    """
    CONFIGURABLE = "configurable"
    THREAD_ID = "thread_id"
    USER_ID = "user_id"
    BOOKING_SERVICE = "booking_service"
    SQL_VIEW_SERVICE = "sql_view_service"
    RAG_SERVICE = "rag_service"


HISTORY_BY_INTENT: dict[str, int] = {
    "booking": 12,
    "cancellation": 6,
    "booking_lookup": 6,
    "search": 6,
    "hybrid_search": 6,
    "document_query": 4,
    "general": 4,
}


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
    TOOLS = "tools"
    CONTEXT_UPDATE = "context_update"
    SAFETY = "safety"
    END = END
