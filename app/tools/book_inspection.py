"""
LangGraph @tool for booking a property inspection via the .NET backend.
Only call this after availability has been checked, a slot chosen,
contact details collected, and the user has explicitly confirmed.
"""
import structlog

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.core.constants import AppStateKeys
from app.core.exceptions import BookingServiceError, ToolValidationError
from app.schemas.booking import BookingRequest, BookingResult
from app.services.booking_service import BookingService

logger = structlog.get_logger(__name__)


@tool
async def book_inspection(slot_id: str, config: RunnableConfig) -> dict:
    """
    Book a property inspection via the .NET backend.
    """
    booking_service: BookingService = config[AppStateKeys.CONFIGURABLE][AppStateKeys.BOOKING_SERVICE]
    user_id = config[AppStateKeys.CONFIGURABLE][AppStateKeys.USER_ID]
    logger.info("book_inspection_start", slot_id=slot_id, user_id=user_id)

    try:
        result = await booking_service.book(
            BookingRequest(slot_id=slot_id, user_id=user_id, notes="")
        )

        logger.info("book_inspection_confirmed", confirmation_id=result.confirmation_id, user_id=user_id)

        return BookingResult(
            success=True,
            confirmation_id=result.confirmation_id,
            property_id=result.property_id,
            property_address=result.property_address,
            start_at_utc=result.start_at_utc,
            end_at_utc=result.end_at_utc,
            agent_name=f"{result.agent_first_name} {result.agent_last_name}".strip(),
            agent_phone=result.agent_phone,
            message=(
                f"Inspection booked for {result.start_at_utc} to {result.end_at_utc}. "
                f"Reference: {result.confirmation_id}. "
                "A confirmation email will be sent to you shortly."
            ),
        ).model_dump()

    except ToolValidationError as exc:
        logger.warning("book_inspection_validation_error", error=str(exc))
        return BookingResult(success=False, error=str(exc)).model_dump()

    except BookingServiceError as exc:
        logger.error("book_inspection_service_error", error=str(exc))
        return BookingResult(success=False, error=str(exc)).model_dump()

    except Exception as exc:
        logger.exception("book_inspection_unexpected_error", error=str(exc))
        return BookingResult(
            success=False,
            error="Booking failed unexpectedly. Please try again or contact the agency directly."
        ).model_dump()
