"""
Constants used across the codebase, e.g. API endpoints, config keys, etc.
"""


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
    SEARCH_LISTINGS = "search_listings"
    SEARCH_DOCUMENTS = "search_documents"
    CHECK_AVAILABILITY = "check_inspection_availability"
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
