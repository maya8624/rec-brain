"""
LangGraph @tool that checks if a holding deposit exists for the current user
against a specific listing. Returns deposit data for the frontend to open
a payment popup, or a not-found result if no deposit exists.
"""
import logging
import uuid

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.core.constants import AppStateKeys
from app.core.exceptions import DepositServiceError, ToolValidationError
from app.schemas.deposit import DepositResult
from app.services.deposit_service import DepositService

logger = logging.getLogger(__name__)


@tool
async def get_deposit(listing_id: str, config: RunnableConfig) -> dict:
    """
    Check if a holding deposit exists for the user on a specific listing.
    Call this when the user wants to pay or check their holding deposit.
    listing_id must be a valid UUID — extract it from [PROPERTY SEARCH RESULTS]
    or the current property context.
    """
    user_id: str = config[AppStateKeys.CONFIGURABLE][AppStateKeys.USER_ID]
    deposit_service: DepositService = config[AppStateKeys.CONFIGURABLE][AppStateKeys.DEPOSIT_SERVICE]

    try:
        uuid.UUID(listing_id)
    except ValueError as exc:
        raise ToolValidationError(
            f"listing_id must be a valid UUID: {exc}") from exc

    try:
        result = await deposit_service.get_my_deposit(listing_id, user_id)
        return result.model_dump()

    except DepositServiceError as exc:
        logger.error("get_deposit | DepositServiceError: %s", exc)
        return DepositResult(success=False, error=str(exc)).model_dump()

    except Exception as exc:
        logger.exception("get_deposit | unexpected error: %s", exc)
        return DepositResult(success=False, error="Could not retrieve deposit. Please try again.").model_dump()
