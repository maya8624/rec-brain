"""
app/tools/book_inspection.py

LangGraph @tool wrapper that calls the .NET backend booking API
via BookingService. Only call this after:
  1. check_inspection_availability has been called
  2. User has chosen a slot from the returned list
  3. All contact details have been collected
  4. User has explicitly confirmed they want to proceed
"""
import logging

from langchain_core.tools import tool

from app.services.booking_services import (
    BookingService,
    BookingServiceError,
    BookingValidationError,
)

logger = logging.getLogger(__name__)

_service: BookingService | None = None


def _get_service() -> BookingService:
    global _service
    if _service is None:
        _service = BookingService()
    return _service


@tool
async def book_inspection(
    property_id: str,
    datetime_slot: str,
    contact_name: str,
    contact_email: str,
    contact_phone: str,
) -> dict:
    """
    Book a property inspection via the agency system.

    STRICT RULES — only call this tool when ALL of the following are true:
    1. check_inspection_availability was already called for this property
    2. datetime_slot is one of the slots returned by check_inspection_availability
    3. You have collected contact_name, contact_email, AND contact_phone from the user
    4. The user has explicitly confirmed they want to proceed with the booking
       (eg said "yes", "confirm", "go ahead", "book it")

    Never assume or invent contact details — always ask the user.
    Never book without explicit confirmation — always summarise details first.

    datetime_slot format: YYYY-MM-DD HH:MM (eg "2025-06-14 10:00")
    contact_phone: Australian format (eg "0412 345 678" or "02 9123 4567")

    Returns:
        success             — bool
        confirmation_id     — .NET booking reference number
        property_address    — confirmed property address
        confirmed_datetime  — confirmed inspection datetime
        agent_name          — assigned agent name
        agent_phone         — agent contact number
        message             — human-readable confirmation summary
        error               — user-friendly error if success is False
    """
    logger.info(
        "book_inspection | property_id=%s | slot=%s | contact=%s",
        property_id, datetime_slot, contact_email,
    )

    try:
        result = await _get_service().book(
            property_id=property_id,
            datetime_slot=datetime_slot,
            contact_name=contact_name,
            contact_email=contact_email,
            contact_phone=contact_phone,
        )

        logger.info("book_inspection | confirmed | id=%s",
                    result.get("confirmation_id"))

        return {
            "success": True,
            "confirmation_id": result.get("confirmation_id", ""),
            "property_address": result.get("property_address", ""),
            "confirmed_datetime": result.get("confirmed_datetime", ""),
            "agent_name": result.get("agent_name", ""),
            "agent_phone": result.get("agent_phone", ""),
            "message": (
                f"Inspection booked for {result.get('confirmed_datetime')}. "
                f"Confirmation sent to {contact_email}. "
                f"Reference: {result.get('confirmation_id')}."
            ),
        }

    except BookingValidationError as e:
        logger.warning("book_inspection | validation error: %s", e)
        return {"success": False, "error": str(e)}

    except BookingServiceError as e:
        logger.error("book_inspection | BookingServiceError: %s", e)
        return {"success": False, "error": str(e)}

    except Exception as e:
        logger.exception("book_inspection | unexpected error: %s", e)
        return {
            "success": False,
            "error": "Booking failed unexpectedly. Please try again or contact the agency directly.",
        }
