"""
LangGraph @tool for property listing search.
question     — provided by the LLM
sql_service  — injected by sql_search_node via InjectedToolArg
"""

import logging
from typing import Annotated

from langchain_core.tools import tool, InjectedToolArg
from app.services.sql_service import SqlAgentError, SqlAgentService

logger = logging.getLogger(__name__)


@tool
async def search_listings(
    question: str,
    sql_service: Annotated[SqlAgentService, InjectedToolArg]
) -> dict:
    """
    Search property listings by location, price, bedrooms, or property type.
    """

    try:
        result = await sql_service.search(question)
        return result

    except SqlAgentError as e:
        logger.error("search_listings | SqlAgentError: %s", e)

        return {
            "success": False,
            "error": "Property search is temporarily unavailable.",
            "output": None,
            "result_count": 0,
        }

    except Exception as e:
        logger.exception("search_listings | unexpected error: %s", e)
        return {
            "success": False,
            "error": "An unexpected error occurred during property search.",
            "output": None,
            "result_count": 0,
        }
