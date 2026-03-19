"""
LangGraph @tool for cancelling a property inspection via the .NET backend.
Requires explicit user confirmation before cancelling.
"""
import logging
from typing import Annotated
from langchain_core.tools import tool, InjectedToolArg

from app.core.exceptions import BookingServiceError, BookingValidationError
from app.schemas.booking import CancellationRequest, CancellationResult
from app.services.booking_service import BookingService

logger = logging.getLogger(__name__)


@tool
async def cancel_inspection(
    confirmation_id: str,
    booking_service: Annotated[BookingService, InjectedToolArg],
    reason: str | None = None,
) -> dict:
    """
    Cancel an existing property inspection booking.
    """

    logger.info(
        "cancel_inspection | confirmation_id=%s | reason=%s",
        confirmation_id, reason,
    )

    try:
        await booking_service.cancel(
            CancellationRequest(
                confirmation_id=confirmation_id,
                reason=reason
            )
        )

        logger.info("cancel_inspection | cancelled | id=%s", confirmation_id)

        return CancellationResult(
            success=True,
            confirmation_id=confirmation_id,
            message=(
                f"Inspection {confirmation_id} has been successfully cancelled. "
                "A cancellation confirmation has been sent to your email."
            ),
        ).model_dump()

    except BookingValidationError as e:
        logger.warning("cancel_inspection | validation error: %s", e)
        return CancellationResult(
            success=False, confirmation_id=confirmation_id, error=str(e)
        ).model_dump()

    except BookingServiceError as e:
        logger.error("cancel_inspection | service error: %s", e)
        return CancellationResult(
            success=False, confirmation_id=confirmation_id, error=str(e)
        ).model_dump()

    except Exception as e:
        logger.exception("cancel_inspection | unexpected error: %s", e)
        return CancellationResult(
            success=False,
            confirmation_id=confirmation_id,
            error="Cancellation failed. Please contact the agency directly.",
        ).model_dump()
