"""
LangGraph @tool for property listing search.
question     — provided by the LLM
sql_service  — injected by sql_search_node via InjectedToolArg
"""
import logging
from typing import Annotated

from langchain_core.tools import tool, InjectedToolArg

from app.schemas.booking import SearchListingResult
from app.services.sql_service import SqlAgentError, SqlViewService

logger = logging.getLogger(__name__)


@tool
async def search_listings(
    question: str,
    sql_service: Annotated[SqlViewService, InjectedToolArg],
) -> dict:
    """
    Search property listings by location, price, bedrooms, or property type.
    """
    try:
        result = await sql_service.search(question)
        return SearchListingResult(**result).model_dump()

    except SqlAgentError as e:
        logger.error("search_listings | SqlAgentError: %s", e)
        return SearchListingResult(
            success=False,
            error="Property search is temporarily unavailable.",
        ).model_dump()

    except Exception as e:
        logger.exception("search_listings | unexpected error: %s", e)
        return SearchListingResult(
            success=False,
            error="An unexpected error occurred during property search.",
        ).model_dump()
