"""
hybrid_search_node — runs SQL listing search and vector document search concurrently.

Responsibility:
    - Resolve SqlViewService and RagService from FastAPI app state
    - Extract question from the latest HumanMessage
    - Run both searches concurrently via asyncio.gather
    - Returns { retrieved_docs: str, search_results: list }

Never calls the LLM — agent_node handles synthesis.
"""

import asyncio
import json
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.agents.state import RealEstateAgentState
from app.core.constants import AppStateKeys, Node, StateKeys
from app.schemas.property import SearchResult
from app.services.rag_service import RagRetriever
from app.services.sql_service import SqlViewService
from app.agents.nodes._base import (
    last_human_message,
    listing_summary,
    resolve_app_service,
    slim_rows,
    vector_payload,
)

logger = logging.getLogger(__name__)


async def hybrid_search_node(state: RealEstateAgentState, config: RunnableConfig) -> dict[str, Any]:
    """
    Runs SQL listing search and vector document search concurrently.
    """
    question = last_human_message(state)
    if not question:
        logger.warning("hybrid_search_node | no human message found")
        return {}

    sql_service: SqlViewService | None = resolve_app_service(
        config, AppStateKeys.SQL_VIEW_SERVICE, Node.HYBRID_SEARCH
    )
    rag_service: RagRetriever | None = resolve_app_service(
        config, AppStateKeys.RAG_SERVICE, Node.HYBRID_SEARCH
    )

    if sql_service is None or rag_service is None:
        return {}

    sql_outcome, vector_outcome = await asyncio.gather(
        sql_service.search_listings(question),
        rag_service.aretrieve(question),
        return_exceptions=True,
    )

    sql = _unwrap_sql(sql_outcome)
    vector = _unwrap_vector(vector_outcome)
    rows = slim_rows(sql.output or [])
    retrieved_docs = _build_retrieved_docs(sql, vector, rows)

    return {
        StateKeys.RETRIEVED_DOCS: retrieved_docs,
        StateKeys.SEARCH_RESULTS: rows,
    }


def _build_retrieved_docs(sql: SearchResult, vector: dict, rows: list) -> str:
    sql_summary = listing_summary(rows) if rows else "No listings found."
    vector_summary = json.dumps({"vector_results": vector}, default=str)

    retrieved_docs = (
        f"[HYBRID SEARCH RESULTS — {sql.result_count} property listing(s) and "
        f"{vector.get('result_count', 0)} document excerpt(s) found. "
        f"Answer the customer using both sources.]\n"
        f"LISTINGS:\n{sql_summary}\n\n"
        f"DOCUMENTS:\n{vector_summary}"
    )

    return retrieved_docs


# ---------------------------------------------------------------------------
# Result unwrappers — handle exceptions from asyncio.gather
# ---------------------------------------------------------------------------

def _unwrap_sql(outcome: Any) -> SearchResult:
    if isinstance(outcome, Exception):
        logger.error("hybrid_search_node | sql failed | %s", outcome)
        return SearchResult(success=False, error=str(outcome))
    return outcome  # SqlViewService already returns a SearchResult


def _unwrap_vector(outcome: Any) -> dict:
    if isinstance(outcome, Exception):
        logger.error("hybrid_search_node | vector failed | %s", outcome)
        return {"success": False, "error": str(outcome)}
    return {"success": True, **vector_payload(outcome)}
