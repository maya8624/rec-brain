"""
hybrid_search_node — runs SQL listing search and vector document search concurrently.

Sets search_results (SQL rows) and retrieved_docs (vector excerpts) independently.
agent_node injects both into the LLM prompt and synthesises the reply.
"""

import asyncio
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.agents.state import RealEstateAgentState
from app.core.constants import AppStateKeys, Node, StateKeys
from app.schemas.property import SearchResult
from app.agents.nodes._base import (
    last_human_message,
    resolve_app_service,
    slim_rows,
    vector_payload,
)
from app.services.rag_service import RagRetriever
from app.services.sql_service import SqlViewService

logger = logging.getLogger(__name__)


async def hybrid_search_node(state: RealEstateAgentState, config: RunnableConfig) -> dict[str, Any]:
    question = last_human_message(state)
    if not question:
        logger.warning("hybrid_search_node | no human message found")
        return {}

    sql_service: SqlViewService = resolve_app_service(
        config, AppStateKeys.SQL_VIEW_SERVICE, Node.HYBRID_SEARCH
    )

    rag_service: RagRetriever = resolve_app_service(
        config, AppStateKeys.RAG_SERVICE, Node.HYBRID_SEARCH
    )

    ctx = state.get(StateKeys.SEARCH_CONTEXT)
    if ctx and (ctx.get("property_id") or ctx.get("location") or ctx.get("address")):
        sql_task = sql_service.search_from_context(ctx)
    else:
        sql_task = sql_service.search_listings(question)

    sql_outcome, vector_outcome = await asyncio.gather(
        sql_task,
        rag_service.aretrieve(question),
        return_exceptions=True,
    )

    sql = _unwrap_sql(sql_outcome)
    vector = _unwrap_vector(vector_outcome)

    return {
        StateKeys.SEARCH_RESULTS: slim_rows(sql.output or []),
        StateKeys.RETRIEVED_DOCS: _build_vector_docs(vector),
    }


def _build_vector_docs(vector: dict) -> str | None:
    if not vector.get("success") or not vector.get("results"):
        return None
    return "\n\n".join(
        f"[excerpt {i + 1}]\n{r['text']}"
        for i, r in enumerate(vector["results"])
    )


def _unwrap_sql(outcome: Any) -> SearchResult:
    if isinstance(outcome, Exception):
        logger.error("hybrid_search_node | sql failed | %s", outcome)
        return SearchResult(success=False, error=str(outcome))
    return outcome


def _unwrap_vector(outcome: Any) -> dict:
    if isinstance(outcome, Exception):
        logger.error("hybrid_search_node | vector failed | %s", outcome)
        return {"success": False, "error": str(outcome)}
    return {"success": True, **vector_payload(outcome)}
