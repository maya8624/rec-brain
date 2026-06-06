"""
LangGraph @tool wrapper that calls the .NET backend availability API
via BookingService. Always call this BEFORE book_inspection
so the user can choose from real available slots.
"""
import structlog
import uuid

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from app.core.constants import AppStateKeys
from app.core.exceptions import BookingServiceError, ToolValidationError
from app.schemas.booking import AvailabilityResult
from app.services.booking_service import BookingService
from app.tools._utils import fmt_dt_sydney

logger = structlog.get_logger(__name__)


@tool
async def check_availability(property_id: str, config: RunnableConfig) -> dict:
    """
    Check available inspection time slots for a property.
    Call immediately once you have the property_id.
    """
    booking_service: BookingService = config[AppStateKeys.CONFIGURABLE][AppStateKeys.BOOKING_SERVICE]

    try:
        uuid.UUID(property_id)
    except ValueError as exc:
        raise ToolValidationError(
            f"property_id must be a valid UUID: {exc}") from exc

    try:
        result = await booking_service.check_availability(property_id)

        logger.info("check_availability_found_slots", slot_count=result.slot_count, property_id=property_id)

        for slot in result.available_slots:
            slot.start_at = fmt_dt_sydney(slot.start_at)
            slot.end_at = fmt_dt_sydney(slot.end_at)

        return result.model_dump()

    except BookingServiceError as exc:
        logger.error("check_availability_service_error", error=str(exc))

        return AvailabilityResult(
            success=False,
            property_id=property_id,
            error=str(exc),
        ).model_dump()

    except Exception as exc:
        logger.exception("check_availability_unexpected_error", error=str(exc))

        return AvailabilityResult(
            success=False,
            property_id=property_id,
            error="Could not retrieve availability. Please try again.",
        ).model_dump()
