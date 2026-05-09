"""
agent_node — the LLM brain of the agent.

Three roles depending on intent:

    1. "booking" / "cancellation" / "booking_lookup" / "deposit_payment"  — first call
       — LLM with tools bound
       — decides which action tool to call 
       (check_availability, book_inspection, cancel_inspection, get_booking, get_deposit)

    2. Any intent — second call (ToolMessage in state)
       — plain LLM, no tools
       — formats tool results into human-readable response

    3. "general" / "search" / "document_query" / "search_then_deposit" / formatting pass
       — plain LLM, no tools
       — responds directly or formats retrieved context
"""

import json
import logging
from typing import Any

from openai import APIStatusError, RateLimitError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agents.nodes._base import format_listings
from app.agents.state import RealEstateAgentState
from app.core.constants import IntentConfig, PromptLabels, StateKeys
from app.infrastructure.llm import get_llm
from app.prompts.agent import REAL_ESTATE_AGENT_SYSTEM, SEARCH_RESULT_SYSTEM
from app.tools import get_all_tools

logger = logging.getLogger(__name__)

# Only these intents need tool calling
_TOOL_INTENTS = frozenset(
    ["booking", "cancellation", "booking_lookup", "deposit_payment"])


async def agent_node(state: RealEstateAgentState) -> dict[str, Any]:
    """
    Primary LLM node.

    First call (HumanMessage is last): uses tool-bound LLM for action intents
        ("booking", "cancellation", "booking_lookup", "deposit_payment")
        so the LLM can invoke tools.

    Second call (ToolMessage is last): uses plain LLM to format the tool result
        into a human-readable reply — no tools bound to prevent re-triggering.

    Non-action intents ("general", "search", "document_query", "hybrid_search",
        "search_then_deposit"):
        always use plain LLM, tools never needed.

    intent_completed is set True only on the formatting pass of a tool-bound flow
        (second call), signalling the next turn's classifier that the intent is done.
        last_intent is updated only then — non-action intents leave it unchanged.
    """
    needs_tools = _needs_tools(state)
    llm = _get_tool_llm() if needs_tools else _get_plain_llm()

    intent = state.get(StateKeys.USER_INTENT, "general")
    history_limit = IntentConfig.HISTORY_BY_INTENT.get(intent, 6)
    history = list(state["messages"])[-history_limit:]

    if needs_tools:
        booking_ctx = state.get(StateKeys.BOOKING_CONTEXT) or {}
        if booking_ctx.get("available_slots"):
            # Slot selection / confirmation: keep plain AIMessages so the LLM knows
            # which slot was already discussed and what it's waiting to confirm.
            # Strip ToolMessages and tool-call AIMessages to avoid OpenAI pairing errors.
            history = [m for m in history if not isinstance(m, ToolMessage) and not (
                isinstance(m, AIMessage) and m.tool_calls)]
        else:
            # First booking pass: only HumanMessages — no prior tool pairs to worry about.
            history = [m for m in history if isinstance(m, HumanMessage)]

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
    # elif needs_tools:
    #     logger.warning("agent_node | needs_tools=True but no tool_calls | intent=%s | prompt_len=%d | response=%s",
    #                    intent, len(prompt), response.content[:200])

    is_format_pass = not needs_tools and intent in _TOOL_INTENTS

    return {
        "messages": [response],
        StateKeys.RETRIEVED_DOCS: None,
        StateKeys.INTENT_COMPLETED: is_format_pass,
        StateKeys.LAST_INTENT: intent,
    }


def _build_prompt(
    state: RealEstateAgentState,
    intent: str,
    history: list,
) -> list:
    # retrieved_docs is injected here and never written to state["messages"],
    # so it cannot accumulate across turns or be mistaken for history.
    system = SEARCH_RESULT_SYSTEM if intent == "search" else REAL_ESTATE_AGENT_SYSTEM
    prompt: list = [SystemMessage(content=system), *history]

    retrieved_docs = state.get(StateKeys.RETRIEVED_DOCS)
    if retrieved_docs:
        prompt.append(SystemMessage(
            content=f"{PromptLabels.RETRIEVED_DOCUMENTS}\n{retrieved_docs}"))

    _search_intents = _TOOL_INTENTS | {
        "search",
        "hybrid_search",
        "search_then_deposit",
    }
    if intent in _search_intents:
        search_results = state.get(StateKeys.SEARCH_RESULTS)
        if search_results:
            label = "" if intent == "search" else f"{PromptLabels.PROPERTY_SEARCH_RESULTS}\n"
            prompt.append(SystemMessage(
                content=f"{label}{format_listings(search_results)}"))
        elif intent == "search":
            prompt.append(SystemMessage(content="No results found."))

    _booking_intents = frozenset(["booking", "cancellation", "booking_lookup"])
    if intent in _booking_intents:
        booking_ctx = state.get(StateKeys.BOOKING_CONTEXT)
        if booking_ctx:
            prompt.append(SystemMessage(
                content=f"{PromptLabels.BOOKING_CONTEXT}\n{json.dumps(booking_ctx, default=str)}"))

    return prompt


def _get_plain_llm():
    return get_llm()


def _get_tool_llm():
    return get_llm().bind_tools(get_all_tools())


def _needs_tools(state: RealEstateAgentState) -> bool:
    """
    Returns True only when:
        - intent is "booking", "cancellation", "booking_lookup", or "deposit_payment"
        - last message is HumanMessage (first call, not formatting pass)
    """
    intent = state.get(StateKeys.USER_INTENT, "general")
    if intent not in _TOOL_INTENTS:
        return False

    if not state["messages"]:
        return False

    return isinstance(state["messages"][-1], HumanMessage)
