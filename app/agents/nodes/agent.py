"""
agent_node — the LLM brain of the agent.

Three roles depending on intent:

    1. "booking" / "cancellation"  — first call
       — uses _LLM_WITH_TOOLS
       — LLM decides which action tool to call (check_availability,
         book_inspection, cancel_inspection)

    2. Any intent — second call (ToolMessage in state)
       — uses _LLM_PLAIN
       — formats tool results into human-readable response

    3. "general" / formatting pass
       — uses _LLM_PLAIN
       — responds directly, no tools needed
"""

import logging
from typing import Any

from groq import APIStatusError, RateLimitError
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.state import RealEstateAgentState
from app.infrastructure.llm import get_llm
from app.prompts.agent import REAL_ESTATE_AGENT_SYSTEM
from app.tools import get_all_tools

logger = logging.getLogger(__name__)

# Only these intents need tool calling
_TOOL_INTENTS = frozenset(["booking", "cancellation"])
_MAX_HISTORY = 10


def _get_plain_llm():
    return get_llm()


def _get_tool_llm():
    return get_llm().bind_tools(get_all_tools())


async def agent_node(state: RealEstateAgentState) -> dict[str, Any]:
    """
    Primary LLM node.

    Uses _LLM_WITH_TOOLS only when:
        - intent is "booking" or "cancellation"
        - last message is HumanMessage (first call, no results yet)

    Uses _LLM_PLAIN for everything else:
        - "general", "search", "document_query"
        - any second call after tool results are back
    """
    needs_tools = _needs_tools(state)
    llm = _get_tool_llm() if needs_tools else _get_plain_llm()

    # TODO: optimize by sending only relevant subset based on call type
    messages = [
        SystemMessage(content=REAL_ESTATE_AGENT_SYSTEM),
        *state["messages"][-_MAX_HISTORY:],
    ]

    logger.info(
        "agent_node | intent=%s | needs_tools=%s | messages=%d | errors=%d",
        state.get("user_intent", "general"),
        needs_tools,
        len(state["messages"]),
        state.get("error_count", 0),
    )

    try:
        response = await llm.ainvoke(messages)
    except RateLimitError as exc:
        logger.error("agent_node | Groq rate limit hit: %s", exc)
        raise
    except APIStatusError as exc:
        logger.error("agent_node | Groq API error %s: %s", exc.status_code, exc.message)
        raise

    if hasattr(response, "tool_calls") and response.tool_calls:
        logger.info(
            "agent_node | tool_calls=%s",
            [tc["name"] for tc in response.tool_calls],
        )

    return {"messages": [response]}


def _needs_tools(state: RealEstateAgentState) -> bool:
    """
    Returns True only when:
        - intent is "booking" or "cancellation"
        - last message is HumanMessage (first call, not formatting pass)
    """
    intent = state.get("user_intent", "general")
    if intent not in _TOOL_INTENTS:
        return False

    if not state["messages"]:
        return False

    return isinstance(state["messages"][-1], HumanMessage)
