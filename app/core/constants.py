"""
Constants used across the codebase, e.g. API endpoints, config keys, etc.
"""


from langgraph.graph import END


class BookingEndpoints:
    """
    Constants for the .NET backend API.
    """
    AVAILABILITY = "/api/inspections/availability"
    BOOK = "/api/inspections/book"
    CANCEL = "/api/inspections/cancel"


class ToolNames:
    """
    Constants for tool names used in the AI agent.
    """
    CHECK_AVAILABILITY = "check_availability"
    BOOK_INSPECTION = "book_inspection"
    CANCEL_INSPECTION = "cancel_inspection"


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
    SEARCH_CONTEXT = "search_context"
    BOOKING_CONTEXT = "booking_context"
    BOOKING_STATUS = "booking_status"


class Node:
    """
    Constants for graph node names.
    """
    INTENT = "intent"
    AGENT = "agent"
    LISTING_SEARCH = "listing_search"
    VECTOR_SEARCH = "vector_search"
    SQL_SEARCH = "sql_search"
    TOOLS = "tools"
    CONTEXT_UPDATE = "context_update"
    SAFETY = "safety"
    END = END
