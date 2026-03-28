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

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.agents.state import RealEstateAgentState
from app.services.vector_search import RagRetriever

logger = logging.getLogger(__name__)


async def vector_search_node(
    state: RealEstateAgentState,
    runnable_config: RunnableConfig,
) -> dict:
    """
    Direct vector search — no tool calls.

    Expects RagRetriever to be available at:
        runnable_config["configurable"]["request"].app.state.rag_retriever

    Returns partial state: { messages: [SystemMessage] }
    Returns {} if there is no HumanMessage or the service cannot be resolved.
    """
    question = _get_last_human_message(state)
    if not question:
        logger.warning("vector_search_node | no human message found")
        return {}

    rag_retriever = _resolve_rag_retriever(runnable_config)
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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_last_human_message(state: RealEstateAgentState) -> str:
    """Return the content of the most recent HumanMessage in state."""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.content if isinstance(message.content, str) else ""
    return ""


def _resolve_rag_retriever(config: RunnableConfig) -> RagRetriever | None:
    """
    Extract RagRetriever from the FastAPI request stored in RunnableConfig.

    Returns None and logs an error rather than raising, so the caller can
    decide how to handle the missing service.
    """
    try:
        request = config.get("configurable", {}).get("request")
        if request is None:
            raise ValueError("no 'request' key in configurable")
        return request.app.state.rag_retriever
    except Exception as exc:
        logger.error(
            "vector_search_node | could not resolve rag_retriever: %s", exc)
        return None
