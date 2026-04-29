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

from app.agents.nodes._base import listing_summary
from app.agents.state import RealEstateAgentState
from app.core.constants import HISTORY_BY_INTENT, StateKeys
from app.infrastructure.llm import get_llm
from app.prompts.agent import REAL_ESTATE_AGENT_SYSTEM
from app.tools import get_all_tools

logger = logging.getLogger(__name__)

# Only these intents need tool calling
_TOOL_INTENTS = frozenset(["booking", "cancellation", "booking_lookup"])


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

    intent = state.get(StateKeys.USER_INTENT, "general")
    history_limit = HISTORY_BY_INTENT.get(intent, 6)
    history = list(state["messages"])[-history_limit:]

    # Build prompt: system + history + current-turn search results (if any).
    # retrieved_docs is injected here and never written to state["messages"],
    # so it cannot accumulate across turns or be mistaken for history.
    prompt: list = [SystemMessage(content=REAL_ESTATE_AGENT_SYSTEM), *history]
    retrieved_docs = state.get(StateKeys.RETRIEVED_DOCS)
    if retrieved_docs:
        prompt.append(SystemMessage(content=retrieved_docs))

    if intent in _TOOL_INTENTS and state.get(StateKeys.SEARCH_RESULTS):
        summary = listing_summary(state[StateKeys.SEARCH_RESULTS])
        prompt.append(SystemMessage(
            content=f"[PROPERTY SEARCH RESULTS]\n{summary}"))

    logger.info(
        "agent_node | intent=%s | needs_tools=%s | history=%d/%d | errors=%d",
        intent,
        needs_tools,
        len(history),
        len(state["messages"]),
        state.get(StateKeys.ERROR_COUNT, 0),
    )

    try:
        response = await llm.ainvoke(prompt)
    except RateLimitError as exc:
        logger.error("agent_node | Groq rate limit hit: %s", exc)
        raise
    except APIStatusError as exc:
        logger.error("agent_node | Groq API error %s: %s",
                     exc.status_code, exc.message)
        raise

    if hasattr(response, "tool_calls") and response.tool_calls:
        logger.info(
            "agent_node | tool_calls=%s",
            [tc["name"] for tc in response.tool_calls],
        )

    # Clear retrieved_docs so the next turn starts clean.
    # Track intent completion for the next turn's classifier.
    is_formatting_pass = not needs_tools and intent in _TOOL_INTENTS
    intent_completed = is_formatting_pass

    # Clear retrieved_docs so the next turn starts clean.
    return {
        "messages": [response],
        StateKeys.RETRIEVED_DOCS:   None,
        StateKeys.INTENT_COMPLETED: intent_completed,
        StateKeys.LAST_INTENT:      intent if intent_completed else state.get(StateKeys.LAST_INTENT),
    }


def _get_plain_llm():
    return get_llm()


def _get_tool_llm():
    return get_llm().bind_tools(get_all_tools())


def _needs_tools(state: RealEstateAgentState) -> bool:
    """
    Returns True only when:
        - intent is "booking" or "cancellation"
        - last message is HumanMessage (first call, not formatting pass)
    """
    intent = state.get(StateKeys.USER_INTENT, "general")
    if intent not in _TOOL_INTENTS:
        return False

    if not state["messages"]:
        return False

    return isinstance(state["messages"][-1], HumanMessage)
