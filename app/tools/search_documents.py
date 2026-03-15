"""
app/tools/search_documents.py

LangGraph @tool wrapper around RagService.
Handles semantic search over property documents —
leases, contracts, strata reports, and property descriptions.
"""
import logging

from langchain_core.tools import tool

from app.services.rag_services import RagService, RagServiceError

logger = logging.getLogger(__name__)

_service: RagService | None = None


def _get_service() -> RagService:
    global _service
    if _service is None:
        _service = RagService()
    return _service


@tool
async def search_documents(
    question: str,
    property_id: str | None = None,
) -> dict:
    """
    Search property documents using semantic search.

    Use this tool when the user asks about:
    - Lease terms        (eg "what is the weekly rent", "notice period")
    - Break lease        (eg "how do I break the lease", "break lease fee")
    - Bond / deposit     (eg "how much is the bond", "when do I get it back")
    - Strata             (eg "strata levy", "by-laws", "pet policy")
    - Contract terms     (eg "settlement date", "cooling off period")
    - Property conditions (eg "what appliances are included")
    - Landlord / tenant obligations

    Optionally pass property_id to restrict search to a specific property's
    documents. If not provided, searches across all documents.

    Examples:
        question="what are the break lease conditions", property_id="prop_123"
        question="are pets allowed"
        question="what is the bond amount for this property", property_id="prop_456"

    Returns:
        success   — bool
        answer    — synthesized answer string for the LLM to present
        sources   — list of source documents with filename, page, relevance score
        error     — user-friendly error message if success is False
    """
    logger.info(
        "search_documents | question=%s | property_id=%s",
        question[:80], property_id,
    )

    try:
        result = await _get_service().search(question, property_id=property_id)
        return result

    except RagServiceError as e:
        logger.error("search_documents | RagServiceError: %s", e)
        return {
            "success": False,
            "error": "Document search is temporarily unavailable. Please try again shortly.",
            "answer": None,
            "sources": [],
        }

    except Exception as e:
        logger.exception("search_documents | unexpected error: %s", e)
        return {
            "success": False,
            "error": "An unexpected error occurred during document search.",
            "answer": None,
            "sources": [],
        }
