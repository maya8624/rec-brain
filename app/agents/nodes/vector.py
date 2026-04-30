"""
vector_search_node — direct vector search via RagService.

Responsibility:
    - Resolve RagService from FastAPI app state via RunnableConfig
    - Extract question from the latest HumanMessage
    - Call RagService.aretrieve
    - Returns { retrieved_docs: str }

Never calls the LLM — agent_node handles synthesis.
"""

import json
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.agents.nodes._base import last_human_message, resolve_app_service, vector_payload
from app.agents.state import RealEstateAgentState
from app.core.constants import AppStateKeys, Node, StateKeys
from app.services.rag_service import RagRetriever

logger = logging.getLogger(__name__)


async def vector_search_node(
        state: RealEstateAgentState,
        config: RunnableConfig) -> dict[str, Any]:
    """
    Direct vector search — no tool calls.
    Returns { retrieved_docs: str }
    """
    question = last_human_message(state)
    if not question:
        logger.warning("vector_search_node | no human message found")
        return {}

    rag_service: RagRetriever | None = resolve_app_service(
        config, AppStateKeys.RAG_SERVICE, Node.VECTOR_SEARCH
    )

    if rag_service is None:
        return {}

    try:
        nodes = await rag_service.aretrieve(question)
        return {StateKeys.RETRIEVED_DOCS: _build_retrieved_docs(nodes)}

    except Exception as exc:
        logger.exception("vector_search_node | failed | %s", exc)
        return {}


def _build_retrieved_docs(nodes: list) -> str:
    count = len(nodes)
    payload = json.dumps(vector_payload(nodes))
    return (
        f"[DOCUMENT SEARCH RESULTS — {count} relevant excerpt(s) found. "
        f"Answer the customer's question using only this retrieved content.]\n{payload}"
    )
