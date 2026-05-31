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

from app.agents.nodes._base import extract_sources, last_human_message, resolve_app_service, vector_payload
from app.agents.state import RealEstateAgentState, RetrievedDocs
from app.agents.nodes.rag_intent import classify_rag_intent
from app.core.constants import AppStateKeys, Node, StateKeys
from app.schemas.rag import INTENT_DOC_TYPES
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

    try:
        rag_service: RagRetriever = resolve_app_service(
            config, AppStateKeys.RAG_SERVICE, Node.VECTOR_SEARCH
        )

        property_id = config.get(AppStateKeys.CONFIGURABLE, {}).get(AppStateKeys.PROPERTY_ID)
        rag_intent = await classify_rag_intent(question)
        doc_types = INTENT_DOC_TYPES.get(rag_intent)
        nodes = await rag_service.aretrieve(query=question, doc_types=doc_types, property_id=property_id)

        payload = json.dumps(vector_payload(nodes))
        docs = (
            f"[DOCUMENT SEARCH RESULTS — {len(nodes)} relevant excerpt(s) found. "
            f"Answer the customer's question using only this retrieved content.]\n{payload}"
        )

        sources = extract_sources(nodes)
        docs = RetrievedDocs(docs=docs, sources=sources)
        result = {StateKeys.RETRIEVED_DOCS: docs}
        return result
    except Exception as exc:
        logger.exception("vector_search_node | failed | %s", exc)
        return {}