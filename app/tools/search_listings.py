"""
app/tools/search_listings.py

LangGraph @tool wrapper around SqlAgentService.
Thin by design — no business logic here.
The LLM reads the docstring to decide when to call this tool,
so it must be precise and example-rich.
"""
import logging

from langchain_core.tools import tool

from app.services.sql_service import SqlAgentError, SqlAgentService

logger = logging.getLogger(__name__)

_service: SqlAgentService | None = None


def _get_service() -> SqlAgentService:
    global _service
    if _service is None:
        _service = SqlAgentService()
    return _service


@tool
def search_listings(question: str) -> dict:
    """
    Search property listings using natural language.

    Use this tool when the user wants to find properties by:
    - Location / suburb  (eg "in Castle Hill", "near Parramatta")
    - Price range        (eg "under $800k", "between $500 and $700 per week")
    - Bedrooms/bathrooms (eg "3 bedroom", "2 bath")
    - Property type      (eg "house", "apartment", "townhouse", "unit")
    - Any combination of the above

    Examples:
        "5 bedroom houses under $1.2M in Bella Vista"
        "2 bedroom apartments for rent under $600 per week in Parramatta"
        "show me townhouses in Castle Hill with 3 bedrooms"

    Returns:
        success      — bool
        output       — formatted property results string for the LLM to present
        sql_used     — the SQL that was executed (for debugging)
        result_count — number of matching properties found
        error        — user-friendly error message if success is False
    """
    logger.info("search_listings | question=%s", question[:80])

    try:
        result = _get_service().search(question)
        return result

    except SqlAgentError as e:
        logger.error("search_listings | SqlAgentError: %s", e)
        return {
            "success": False,
            "error": "Property search is temporarily unavailable. Please try again shortly.",
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
