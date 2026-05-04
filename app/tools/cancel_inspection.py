"""
LangGraph @tool for cancelling a property inspection via the .NET backend.
Requires explicit user confirmation before cancelling.
"""
import logging
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.core.constants import AppStateKeys
from app.core.exceptions import BookingServiceError, ToolValidationError
from app.schemas.booking import CancellationResult
from app.services.booking_service import BookingService

logger = logging.getLogger(__name__)


@tool
async def cancel_inspection(confirmation_id: str, config: RunnableConfig, reason: str | None = None) -> dict:
    """
    Cancel an existing property inspection booking.
    """
    booking_service: BookingService = config[AppStateKeys.CONFIGURABLE][AppStateKeys.BOOKING_SERVICE]
    user_id: str = config[AppStateKeys.CONFIGURABLE][AppStateKeys.USER_ID]

    logger.info(
        "cancel_inspection | confirmation_id=%s | reason=%s",
        confirmation_id, reason,
    )

    try:
        result = await booking_service.cancel(confirmation_id, user_id)
        logger.info("cancel_inspection | cancelled | id=%s", confirmation_id)
        return result.model_dump()

    except ToolValidationError as exc:
        logger.warning("cancel_inspection | validation error: %s", exc)
        return CancellationResult(
            success=False, id=confirmation_id, error=str(exc)
        ).model_dump()
    except BookingServiceError as exc:
        logger.error("cancel_inspection | service error: %s", exc)
        return CancellationResult(
            success=False, id=confirmation_id, error=str(exc)
        ).model_dump()
    except Exception as exc:
        logger.exception("cancel_inspection | unexpected error: %s", exc)
        return CancellationResult(
            success=False,
            id=confirmation_id,
            error="Cancellation failed. Please contact the agency directly.",
        ).model_dump()
