"""
app.agents.nodes.vector
~~~~~~~~~~~~~~~~~~~~~~~
vector_search_node — direct vector search via RagRetriever.

Responsibility:
    - Resolve RagRetriever from FastAPI app state via RunnableConfig
    - Extract question from the latest HumanMessage
    - Call RagRetriever.aretrieve
    - Store results as SystemMessage for agent_node to format
    - Return {"messages": [SystemMessage]}

Never calls the LLM — agent_node handles synthesis.
"""

import json
import logging

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from app.agents.nodes._base import last_human_message, resolve_app_service
from app.agents.state import RealEstateAgentState
from app.core.constants import AppStateKeys
from app.services.rag_service import RagRetriever

logger = logging.getLogger(__name__)


async def vector_search_node(
    state: RealEstateAgentState,
    config: RunnableConfig,
) -> dict:
    """
    Direct vector search — no tool calls.

    Expects RagRetriever to be available at:
        config["configurable"]["request"].app.state.rag_retriever

    Returns partial state: { messages: [SystemMessage] }
    Returns {} if there is no HumanMessage or the service cannot be resolved.
    """
    question = last_human_message(state)
    if not question:
        logger.warning("vector_search_node | no human message found")
        return {}

    rag_retriever: RagRetriever | None = resolve_app_service(
        config, AppStateKeys.RAG_RETRIEVER, "vector_search_node"
    )
    if rag_retriever is None:
        return {}

    try:
        nodes = await rag_retriever.aretrieve(question)

        logger.info("vector_search_node | retrieved %d nodes", len(nodes))

        result_message = SystemMessage(content=json.dumps({
            "results": [
                {
                    "text": n.node.get_content(),
                    "score": n.score,
                    "metadata": n.node.metadata,
                }
                for n in nodes
            ],
            "result_count": len(nodes),
            "source": "vector_db",
        }))

        return {"messages": [result_message]}

    except Exception as exc:
        logger.exception("vector_search_node | failed | %s", exc)
        return {}
