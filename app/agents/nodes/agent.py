"""
agent_node — the LLM brain of the agent.

Three roles depending on intent:

    1. "booking" / "cancellation" / "booking_lookup"  — first call
       — LLM with tools bound
       — decides which action tool to call 
       (check_availability, book_inspection, cancel_inspection, get_booking)

    2. Any intent — second call (ToolMessage in state)
       — plain LLM, no tools
       — formats tool results into human-readable response

    3. "general" / "search" / "document_query" / formatting pass
       — plain LLM, no tools
       — responds directly or formats retrieved context
"""

import logging
from typing import Any

from openai import APIStatusError, RateLimitError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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

    First call (HumanMessage is last): uses tool-bound LLM for action intents
        ("booking", "cancellation", "booking_lookup") so the LLM can invoke tools.

    Second call (ToolMessage is last): uses plain LLM to format the tool result
        into a human-readable reply — no tools bound to prevent re-triggering.

    Non-action intents ("general", "search", "document_query", "hybrid_search"):
        always use plain LLM, tools never needed.

    intent_completed is set True only on the formatting pass of a tool-bound flow
        (second call), signalling the next turn's classifier that the intent is done.
        last_intent is updated only then — non-action intents leave it unchanged.
    """
    needs_tools = _needs_tools(state)
    llm = _get_tool_llm() if needs_tools else _get_plain_llm()

    intent = state.get(StateKeys.USER_INTENT, "general")
    history_limit = HISTORY_BY_INTENT.get(intent, 6)
    history = list(state["messages"])[-history_limit:]

    prompt = _build_prompt(state, intent, history)

    try:
        response: AIMessage = await llm.ainvoke(prompt)
    except RateLimitError as exc:
        logger.error("agent_node | OpenAI rate limit hit: %s", exc)
        raise
    except APIStatusError as exc:
        logger.error("agent_node | OpenAI API error %s: %s",
                     exc.status_code, exc.message)
        raise

    if response.tool_calls:
        logger.info("agent_node | tool_calls=%s", [
                    tc["name"] for tc in response.tool_calls])

    intent_completed = not needs_tools and intent in _TOOL_INTENTS
    last_intent = intent if intent_completed else state.get(
        StateKeys.LAST_INTENT)

    return {
        "messages": [response],
        StateKeys.RETRIEVED_DOCS: None,
        StateKeys.INTENT_COMPLETED: intent_completed,
        StateKeys.LAST_INTENT: last_intent,
    }


def _build_prompt(
    state: RealEstateAgentState,
    intent: str,
    history: list,
) -> list:
    # retrieved_docs is injected here and never written to state["messages"],
    # so it cannot accumulate across turns or be mistaken for history.
    prompt: list = [SystemMessage(content=REAL_ESTATE_AGENT_SYSTEM), *history]

    retrieved_docs = state.get(StateKeys.RETRIEVED_DOCS)
    if retrieved_docs:
        prompt.append(SystemMessage(
            content=f"[RETRIEVED DOCUMENTS]\n{retrieved_docs}"))

    if intent in _TOOL_INTENTS and state.get(StateKeys.SEARCH_RESULTS):
        summary = listing_summary(state[StateKeys.SEARCH_RESULTS])
        prompt.append(SystemMessage(
            content=f"[PROPERTY SEARCH RESULTS]\n{summary}"))

    return prompt


def _get_plain_llm():
    return get_llm()


def _get_tool_llm():
    return get_llm().bind_tools(get_all_tools())


def _needs_tools(state: RealEstateAgentState) -> bool:
    """
    Returns True only when:
        - intent is "booking", "cancellation", or "booking_lookup"
        - last message is HumanMessage (first call, not formatting pass)
    """
    intent = state.get(StateKeys.USER_INTENT, "general")
    if intent not in _TOOL_INTENTS:
        return False

    if not state["messages"]:
        return False

    return isinstance(state["messages"][-1], HumanMessage)
