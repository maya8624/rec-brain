"""
app/tools/cancel_inspection.py

LangGraph @tool wrapper that calls the .NET backend cancellation API
via BookingService. Requires explicit user confirmation before cancelling
— the same rule as booking.
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
async def cancel_inspection(
    confirmation_id: str,
    reason: str | None = None,
) -> dict:
    """
    Cancel an existing property inspection booking.

    Only call this tool when:
    1. You have the confirmation_id of the booking to cancel
    2. The user has explicitly confirmed they want to cancel
       (eg said "yes cancel it", "go ahead", "please cancel")

    Always confirm the booking details with the user before cancelling.
    Never cancel without explicit user confirmation.

    confirmation_id: the booking reference number (eg "CONF-12345")
    reason: optional cancellation reason from the user

    Returns:
        success         — bool
        confirmation_id — echoed back for context_update_node
        message         — human-readable cancellation confirmation
        error           — user-friendly error if success is False
    """
    logger.info(
        "cancel_inspection | confirmation_id=%s | reason=%s",
        confirmation_id, reason,
    )

    try:
        await _get_service().cancel(
            confirmation_id=confirmation_id,
            reason=reason,
        )

        logger.info("cancel_inspection | cancelled | id=%s", confirmation_id)

        return {
            "success": True,
            "confirmation_id": confirmation_id,
            "message": (
                f"Inspection {confirmation_id} has been successfully cancelled. "
                "A cancellation confirmation has been sent to your email."
            ),
        }

    except BookingValidationError as e:
        logger.warning("cancel_inspection | validation error: %s", e)
        return {"success": False, "confirmation_id": confirmation_id, "error": str(e)}

    except BookingServiceError as e:
        logger.error("cancel_inspection | BookingServiceError: %s", e)
        return {"success": False, "confirmation_id": confirmation_id, "error": str(e)}

    except Exception as e:
        logger.exception("cancel_inspection | unexpected error: %s", e)
        return {
            "success": False,
            "confirmation_id": confirmation_id,
            "error": "Cancellation failed. Please contact the agency directly.",
        }
