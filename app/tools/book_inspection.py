"""
LangGraph @tool for booking a property inspection via the .NET backend.
Only call this after availability has been checked, a slot chosen,
contact details collected, and the user has explicitly confirmed.
"""
import logging
from typing import Annotated

from langchain_core.tools import tool
from langchain_core.tools import InjectedToolArg

from app.core.exceptions import BookingServiceError, BookingValidationError
from app.schemas.booking import BookingRequest, BookingResult, ContactInfo
from app.services.booking_service import BookingService

logger = logging.getLogger(__name__)


@tool
async def book_inspection(
    property_id: str,
    datetime_slot: str,
    contact_name: str,
    contact_email: str,
    contact_phone: str,
    booking_service: Annotated[BookingService, InjectedToolArg]
) -> dict:
    """
    Book a property inspection via the .NET backend.
    """

    logger.info(
        "book_inspection | property_id=%s | slot=%s | contact=%s",
        property_id, datetime_slot, contact_email,
    )

    try:
        contact = ContactInfo(
            name=contact_name,
            email=contact_email,
            phone=contact_phone
        )

        result = await booking_service.book(
            BookingRequest(
                property_id=property_id,
                datetime_slot=datetime_slot,
                contact=contact
            )
        )

        logger.info("book_inspection | confirmed | id=%s",
                    result["confirmation_id"])

        return BookingResult(
            success=True,
            confirmation_id=result["confirmation_id"],
            property_address=result["property_address"],
            confirmed_datetime=result["confirmed_datetime"],
            agent_name=result["agent_name"],
            agent_phone=result["agent_phone"],
            message=(
                f"Inspection booked for {result['confirmed_datetime']}. "
                f"Confirmation sent to {contact_email}. "
                f"Reference: {result['confirmation_id']}."
            ),
        ).model_dump()

    except BookingValidationError as e:
        logger.warning("book_inspection | validation error: %s", e)
        return BookingResult(success=False, error=str(e)).model_dump()

    except BookingServiceError as e:
        logger.error("book_inspection | service error: %s", e)
        return BookingResult(success=False, error=str(e)).model_dump()

    except Exception as e:
        logger.exception("book_inspection | unexpected error: %s", e)
        return BookingResult(
            success=False,
            error="Booking failed unexpectedly. Please try again or contact the agency directly."
        ).model_dump()
