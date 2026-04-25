"""
LangGraph @tool wrapper that calls the .NET backend availability API
via BookingService. Always call this BEFORE book_inspection
so the user can choose from real available slots.
"""
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from app.core.constants import AppStateKeys
from app.core.exceptions import BookingServiceError
from app.schemas.booking import AvailabilityResult
from app.services.booking_service import BookingService

logger = logging.getLogger(__name__)

_SYDNEY = ZoneInfo("Australia/Sydney")


def _fmt_dt(dt_str: str) -> str:
    if not dt_str:
        return "TBD"
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(_SYDNEY).strftime("%a %d %b %Y at %I:%M %p %Z")
    except ValueError:
        return dt_str


@tool
async def check_availability(property_id: str, config: RunnableConfig) -> dict:
    """
    Check available inspection time slots for a property.
    Call immediately once you have the property_id.
    """
    booking_service: BookingService = config["configurable"][AppStateKeys.BOOKING_SERVICE]

    try:
        result = await booking_service.check_availability(property_id)

        logger.info(
            "check_availability | found %d slots for %s",
            result.slot_count, property_id,
        )

        for slot in result.available_slots:
            slot.start_at = _fmt_dt(slot.start_at)
            slot.end_at = _fmt_dt(slot.end_at)

        return result.model_dump()

    except BookingServiceError as exc:
        logger.error("check_availability | BookingServiceError: %s", exc)

        return AvailabilityResult(
            success=False,
            property_id=property_id,
            error=str(exc),
        ).model_dump()

    except Exception as exc:
        logger.exception("check_availability | unexpected error: %s", exc)

        return AvailabilityResult(
            success=False,
            property_id=property_id,
            error="Could not retrieve availability. Please try again.",
        ).model_dump()
