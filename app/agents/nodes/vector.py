"""
app.agents.nodes.vector
~~~~~~~~~~~~~~~~~~~~~~~
vector_search_node — handles search_documents tool calls via Chroma.

Responsibility:
    - Find every search_documents tool call on the latest AIMessage
    - Call perform_vector_search for each one
    - Wrap each result as a ToolMessage (JSON-serialised)
    - Return {"messages": [ToolMessage, ...]}

Never calls the LLM — agent_node handles synthesis.
"""

import logging

from app.agents.nodes._base import build_tool_message, error_content, last_ai_message
from app.agents.state import RealEstateAgentState
from app.services.vector_search import perform_vector_search

logger = logging.getLogger(__name__)

_TOOL_NAME = "search_documents"


async def vector_search_node(state: RealEstateAgentState) -> dict:
    """
    Process all search_documents tool calls on the latest AIMessage.

    Returns partial state: { messages: [ToolMessage, ...] }
    Returns {} if there is no AIMessage or no matching tool calls.
    """
    last_ai = last_ai_message(state)
    if not last_ai:
        return {}

    tool_calls = [tc for tc in last_ai.tool_calls if tc["name"] == _TOOL_NAME]
    if not tool_calls:
        return {}

    tool_messages = []

    for tc in tool_calls:
        question = tc["args"].get("question", "")
        tool_call_id = tc["id"]

        logger.info("vector_search_node | question=%.80s", question)

        try:
            result = await perform_vector_search(question)
            content = (
                {
                    "success": True,
                    "answer": result.get("answer", ""),
                    "source": result.get("source", "vector_db"),
                }
                if result
                else {"success": False, "error": "No results found"}
            )
        except Exception as exc:
            logger.exception(
                "vector_search_node | failed | question=%.80s", question)
            content = error_content(exc)

        tool_messages.append(build_tool_message(
            tool_call_id, _TOOL_NAME, content))

    return {"messages": tool_messages}
