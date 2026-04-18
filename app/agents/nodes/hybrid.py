"""
app.agents.nodes.hybrid_search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
hybrid_search_node — runs SQL listing search and vector document search
concurrently, then stores combined results for agent_node to format.

Responsibility:
    - Resolve SqlViewService and RagRetriever from FastAPI app state
    - Extract question from the latest HumanMessage
    - Run both searches concurrently via asyncio.gather
    - Store combined results as a SystemMessage for agent_node to format
    - Return {"messages": [SystemMessage]}

Never calls the LLM — agent_node handles synthesis.
"""

import asyncio
import json
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.agents.nodes._base import last_human_message, listing_summary, resolve_app_service, slim_rows
from app.agents.state import RealEstateAgentState
from app.core.constants import AppStateKeys, Node

logger = logging.getLogger(__name__)


async def hybrid_search_node(state: RealEstateAgentState, config: RunnableConfig) -> dict[str, Any]:
    """
    Runs SQL listing search and vector document search concurrently.

    Expects services at:
        config["configurable"]["sql_view_service"]
        config["configurable"]["rag_retriever"]

    Returns partial state: { messages: [SystemMessage] }
    Returns {} if there is no HumanMessage or either service cannot be resolved.
    """
    question = last_human_message(state)
    if not question:
        logger.warning("hybrid_search_node | no human message found")
        return {}

    sql_service = resolve_app_service(
        config, AppStateKeys.SQL_VIEW_SERVICE, Node.HYBRID_SEARCH
    )
    rag_retriever = resolve_app_service(
        config, AppStateKeys.RAG_RETRIEVER, Node.HYBRID_SEARCH
    )

    if sql_service is None or rag_retriever is None:
        return {}

    logger.info("hybrid_search_node | question=%.80s", question)

    sql_outcome, vector_outcome = await asyncio.gather(
        sql_service.search_listings(question),
        rag_retriever.aretrieve(question),
        return_exceptions=True,
    )

    sql = _unwrap_sql(sql_outcome)
    vector = _unwrap_vector(vector_outcome)
    sql_count = sql.get("result_count", 0)
    vector_count = vector.get("result_count", 0)

    rows = slim_rows(sql.get("output") or [])
    summary = listing_summary(rows) if rows else "No listings found."
    vector_payload = json.dumps({"vector_results": vector}, default=str)

    retrieved_docs = (
        f"[HYBRID SEARCH RESULTS — {sql_count} property listing(s) and "
        f"{vector_count} document excerpt(s) found. "
        f"Answer the customer using both sources.]\n"
        f"LISTINGS:\n{summary}\n\n"
        f"DOCUMENTS:\n{vector_payload}"
    )

    return {
        "retrieved_docs": retrieved_docs,
        "search_results": rows,
    }


# ---------------------------------------------------------------------------
# Result unwrappers — handle exceptions from asyncio.gather
# ---------------------------------------------------------------------------

def _unwrap_sql(outcome: Any) -> dict:
    if isinstance(outcome, Exception):
        logger.error("hybrid_search_node | sql failed | %s", outcome)
        return {"success": False, "error": str(outcome)}
    return outcome  # SqlViewService already returns a well-formed dict


def _unwrap_vector(outcome: Any) -> dict:
    if isinstance(outcome, Exception):
        logger.error("hybrid_search_node | vector failed | %s", outcome)
        return {"success": False, "error": str(outcome)}
    return {
        "success": True,
        "results": [
            {
                "text": n.node.get_content(),
                "score": n.score,
                "metadata": n.node.metadata,
            }
            for n in outcome
        ],
        "result_count": len(outcome),
    }
