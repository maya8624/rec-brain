"""
app.agents.nodes.vector
~~~~~~~~~~~~~~~~~~~~~~~
vector_search_node — handles search_documents tool calls via RagRetriever.

Responsibility:
    - Resolve RagRetriever from FastAPI app state via RunnableConfig
    - Find every search_documents tool call on the latest AIMessage
    - Call RagRetriever.aretrieve for each one
    - Wrap each result as a ToolMessage (JSON-serialised)
    - Return {"messages": [ToolMessage, ...]}

Never calls the LLM — agent_node handles synthesis.
"""

import logging

from langchain_core.runnables import RunnableConfig

from app.agents.nodes._base import build_tool_message, error_content, last_ai_message
from app.agents.state import RealEstateAgentState
from app.services.vector_search import RagRetriever

logger = logging.getLogger(__name__)

_TOOL_NAME = "search_documents"


async def vector_search_node(
    state: RealEstateAgentState,
    runnable_config: RunnableConfig,
) -> dict:
    """
    Process all search_documents tool calls on the latest AIMessage.

    Expects RagRetriever to be available at:
        runnable_config["configurable"]["request"].app.state.rag_retriever

    Returns partial state: { messages: [ToolMessage, ...] }
    Returns {} if there is no AIMessage, no matching tool calls, or the
    service cannot be resolved.
    """
    last_ai = last_ai_message(state)
    if not last_ai:
        return {}

    tool_calls = [tc for tc in last_ai.tool_calls if tc["name"] == _TOOL_NAME]
    if not tool_calls:
        return {}

    rag_retriever = _resolve_rag_retriever(runnable_config)
    if rag_retriever is None:
        return {}

    tool_messages = []

    for tc in tool_calls:
        question = tc["args"].get("question", "")
        tool_call_id = tc["id"]

        logger.info("vector_search_node | question=%.80s", question)

        try:
            nodes = await rag_retriever.aretrieve(question)
            content = {
                "success": True,
                "results": [
                    {
                        "text": n.node.get_content(),
                        "score": n.score,
                        "metadata": n.node.metadata,
                    }
                    for n in nodes
                ],
                "source": "vector_db",
            }
        except Exception as exc:
            logger.exception(
                "vector_search_node | failed | question=%.80s", question)
            content = error_content(exc)

        tool_messages.append(build_tool_message(
            tool_call_id, _TOOL_NAME, content))

    return {"messages": tool_messages}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

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
    except Exception as e:
        logger.error(
            "vector_search_node | could not resolve rag_retriever: %s", e)
        return None
