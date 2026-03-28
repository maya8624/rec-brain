"""
LangGraph @tool wrapper that calls the .NET backend availability API
via BookingService. Always call this BEFORE book_inspection
so the user can choose from real available slots.
"""
import logging
from typing import Annotated
from langchain_core.tools import InjectedToolArg, tool
from app.core.exceptions import BookingServiceError
from app.schemas.booking import AvailabilityResult, AvailableSlot
from app.services.booking_service import BookingService

logger = logging.getLogger(__name__)


@tool
async def check_availability(
    property_id: str,
    booking_service: Annotated[BookingService, InjectedToolArg],
    preferred_date: str | None = None,
) -> dict:
    """
     Check available inspection time slots for a property.
    preferred_date format: YYYY-MM-DD (eg "2025-06-14")
    """

    logger.info(
        "check_availability | property_id=%s | preferred_date=%s",
        property_id, preferred_date,
    )

    try:
        slots = await booking_service.get_availability(property_id, preferred_date)
        avaliable_slots = [
            AvailableSlot(
                datetime=slot["datetime"],
                agent_name=slot.get("agent_name", ""),
                available=slot.get("available", True),
            )
            for slot in slots if slot.get("datetime")
        ]

        logger.info(
            "check_availability | found %d slots for %s",
            len(avaliable_slots), property_id,
        )

        return AvailabilityResult(
            success=True,
            property_id=property_id,
            available_slots=avaliable_slots,
            slot_count=len(avaliable_slots),
        ).model_dump()

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
