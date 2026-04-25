"""
LangGraph @tool for fetching existing inspection booking details from the .NET backend.
"""
import logging

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.core.constants import AppStateKeys
from app.core.exceptions import BookingServiceError
from app.schemas.booking import BookingConfirmation, BookingLookupResult
from app.services.booking_service import BookingService
from app.tools._utils import fmt_dt_sydney

logger = logging.getLogger(__name__)


def _to_result(booking: BookingConfirmation) -> BookingLookupResult:
    return BookingLookupResult(
        success=True,
        confirmation_id=booking.confirmation_id,
        property_id=booking.property_id,
        property_address=booking.property_address,
        status=booking.status,
        agent_name=f"{booking.agent_first_name} {booking.agent_last_name}".strip(),
        agent_phone=booking.agent_phone,
        start_at=fmt_dt_sydney(booking.start_at_utc),
        end_at=fmt_dt_sydney(booking.end_at_utc),
    )


@tool
async def get_booking(config: RunnableConfig, confirmation_id: str = "") -> dict:
    """
    Retrieve existing inspection booking details for the current user.
    - Call with confirmation_id when the user provides their reference number.
    - Call with no arguments to retrieve all of the user's bookings.
    """
    booking_service: BookingService = config["configurable"][AppStateKeys.BOOKING_SERVICE]
    user_id: str = config["configurable"][AppStateKeys.USER_ID]

    try:
        if confirmation_id:
            booking = await booking_service.get_booking(confirmation_id, user_id)
            return _to_result(booking).model_dump()

        bookings = await booking_service.get_my_bookings(user_id)
        if not bookings:
            return BookingLookupResult(
                success=False,
                error="No bookings found for your account.",
            ).model_dump()

        return BookingLookupResult(
            success=True,
            bookings=[_to_result(booking).model_dump()
                      for booking in bookings],
        ).model_dump()

    except BookingServiceError as exc:
        logger.error("get_booking | service error: %s", exc)
        return BookingLookupResult(success=False, error=str(exc)).model_dump()

    except Exception as exc:
        logger.exception("get_booking | unexpected error: %s", exc)
        return BookingLookupResult(
            success=False,
            error="Could not retrieve booking details. Please try again.",
        ).model_dump()
