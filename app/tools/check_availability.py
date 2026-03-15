"""
app/tools/check_availability.py

LangGraph @tool wrapper that calls the .NET backend availability API
via BookingService. Always call this BEFORE book_inspection
so the user can choose from real available slots.
"""
import logging

from langchain_core.tools import tool

from app.services.booking_service import BookingService, BookingServiceError

logger = logging.getLogger(__name__)

_service: BookingService | None = None


def _get_service() -> BookingService:
    global _service
    if _service is None:
        _service = BookingService()
    return _service


@tool
async def check_availability(
    property_id: str,
    preferred_date: str | None = None,
) -> dict:
    """
    Check available inspection time slots for a property.

    ALWAYS call this before book_inspection so the user can choose
    from real available slots returned by the agency system.
    Never invent or assume time slots — only book slots from this list.

    preferred_date format: YYYY-MM-DD (eg "2025-06-14")
    If preferred_date is not provided, returns next 7 days of availability.

    Examples:
        property_id="prop_123"
        property_id="prop_456", preferred_date="2025-06-14"

    Returns:
        success         — bool
        property_id     — echoed back for context_update_node
        available_slots — list of datetime strings eg ["2025-06-14 10:00", "2025-06-14 14:00"]
        slot_count      — number of available slots
        error           — user-friendly error if success is False
    """
    logger.info(
        "check_availability | property_id=%s | preferred_date=%s",
        property_id, preferred_date,
    )

    try:
        slots = await _get_service().get_availability(
            property_id=property_id,
            preferred_date=preferred_date,
        )

        slot_datetimes = [s["datetime"] for s in slots if s.get("datetime")]

        logger.info(
            "check_availability | found %d slots for %s",
            len(slot_datetimes), property_id,
        )

        return {
            "success": True,
            "property_id": property_id,
            "available_slots": slot_datetimes,
            "slot_count": len(slot_datetimes),
        }

    except BookingServiceError as e:
        logger.error("check_availability | BookingServiceError: %s", e)
        return {
            "success": False,
            "property_id": property_id,
            "available_slots": [],
            "slot_count": 0,
            "error": str(e),
        }

    except Exception as e:
        logger.exception("check_availability | unexpected error: %s", e)
        return {
            "success": False,
            "property_id": property_id,
            "available_slots": [],
            "slot_count": 0,
            "error": "Could not retrieve availability. Please try again.",
        }
