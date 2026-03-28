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

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from app.agents.nodes._base import last_human_message, resolve_app_service
from app.agents.state import RealEstateAgentState

logger = logging.getLogger(__name__)


async def hybrid_search_node(
    state: RealEstateAgentState,
    runnable_config: RunnableConfig,
) -> dict[str, Any]:
    """
    Runs SQL listing search and vector document search concurrently.

    Expects services at:
        runnable_config["configurable"]["request"].app.state.sql_view_service
        runnable_config["configurable"]["request"].app.state.rag_retriever

    Returns partial state: { messages: [SystemMessage] }
    Returns {} if there is no HumanMessage or either service cannot be resolved.
    """
    question = last_human_message(state)
    if not question:
        logger.warning("hybrid_search_node | no human message found")
        return {}

    sql_service = resolve_app_service(
        runnable_config, "sql_view_service", "hybrid_search_node"
    )
    rag_retriever = resolve_app_service(
        runnable_config, "rag_retriever", "hybrid_search_node"
    )

    if sql_service is None or rag_retriever is None:
        return {}

    logger.info("hybrid_search_node | question=%.80s", question)

    sql_outcome, vector_outcome = await asyncio.gather(
        sql_service.search_listings(question),
        rag_retriever.aretrieve(question),
        return_exceptions=True,
    )

    result_message = SystemMessage(content=json.dumps({
        "sql_results": _unwrap_sql(sql_outcome),
        "vector_results": _unwrap_vector(vector_outcome),
    }))

    return {"messages": [result_message]}


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
