"""
agent_node — the LLM brain of the agent.

Three roles:
    1. Tool-calling pass  (action intent + HumanMessage last):
       tool-bound LLM picks check_availability / book_inspection /
       cancel_inspection / get_booking / get_deposit
    2. Formatting pass    (action intent + ToolMessage last):
       plain LLM narrates the tool result — no tools bound
    3. Non-action intents (general / search / document_query / etc.):
       plain LLM responds or formats retrieved context directly
"""

import json
import logging
from typing import Any

from openai import APIStatusError, RateLimitError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agents.nodes._base import format_listings
from app.agents.state import BookingContext, ConversationPhase, RealEstateAgentState, UserIntent
from app.core.constants import IntentConfig, PromptLabels, StateKeys
from app.infrastructure.llm import get_llm
from app.prompts.agent import REAL_ESTATE_AGENT_SYSTEM, SEARCH_RESULT_SYSTEM
from app.tools import get_all_tools

logger = logging.getLogger(__name__)


async def agent_node(state: RealEstateAgentState) -> dict[str, Any]:
    """Routes to tool-bound or plain LLM based on intent and message type, then updates state."""
    needs_tools = _needs_tools(state)
    llm = _get_tool_llm() if needs_tools else _get_plain_llm()
    intent = state.get(StateKeys.USER_INTENT, "general")
    history_limit = IntentConfig.HISTORY_BY_INTENT.get(intent, 6)
    history = list(state["messages"])[-history_limit:]

    if needs_tools:
        booking_ctx = state.get(StateKeys.BOOKING_CONTEXT) or {}
        history = _filter_booking_messages(booking_ctx, history)

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

    state_keys = _update_state_keys(intent, state, response, needs_tools)
    return state_keys


def _filter_booking_messages(
        booking_ctx: BookingContext,
        history: list) -> list[HumanMessage | AIMessage]:

    if booking_ctx.get("available_slots"):
        history = [
            msg for msg in history
            if not isinstance(msg, ToolMessage) and
            not (isinstance(msg, AIMessage) and msg.tool_calls)
        ]
    else:
        history = [msg for msg in history if isinstance(msg, HumanMessage)]

    return history


def _update_state_keys(
        intent: UserIntent,
        state: RealEstateAgentState,
        response: AIMessage,
        needs_tools: bool) -> dict[str, Any]:

    is_complete = (
        intent not in IntentConfig.TOOL_INTENTS
        or (not needs_tools and intent in IntentConfig.TOOL_INTENTS)
    )

    state_keys = {
        "messages":                 [response],
        StateKeys.RETRIEVED_DOCS:   None,
        StateKeys.INTENT_COMPLETED: is_complete
    }

    phase = state.get(StateKeys.PHASE, ConversationPhase.IDLE)

    # User said no to cancellation — reset phase to IDLE
    if intent == "general" and phase == ConversationPhase.CANCELLATION_PENDING:
        state_keys.update({
            StateKeys.PHASE: ConversationPhase.IDLE,
            StateKeys.INTENT_COMPLETED: True,
        })
    # Agent asked user to confirm cancellation — waiting for yes/no
    elif intent == "cancellation" and not response.tool_calls:
        state_keys.update({
            StateKeys.PHASE: ConversationPhase.CANCELLATION_PENDING,
            StateKeys.INTENT_COMPLETED: False,
        })

    return state_keys


def _build_prompt(
        state: RealEstateAgentState,
        intent: str,
        history: list) -> list:

    system = SEARCH_RESULT_SYSTEM if intent == "search" else REAL_ESTATE_AGENT_SYSTEM

    docs_msg = _get_retrieved_docs_msg(state.get(
        StateKeys.RETRIEVED_DOCS)) if intent in IntentConfig.DOC_INTENTS else None

    search_msg = _get_search_msg(intent, state.get(StateKeys.SEARCH_RESULTS) or [
    ]) if intent in IntentConfig.SEARCH_INTENTS else None

    booking_msg = _get_booking_ctx_msg(state.get(
        StateKeys.BOOKING_CONTEXT)) if intent in IntentConfig.BOOKING_INTENTS else None

    blocks = [
        SystemMessage(content=system),
        *history,
        docs_msg,
        search_msg,
        booking_msg
    ]

    return [block for block in blocks if block is not None]


def _get_booking_ctx_msg(booking_ctx: BookingContext) -> SystemMessage | None:
    if not booking_ctx:
        return None
    return SystemMessage(
        content=f"{PromptLabels.BOOKING_CONTEXT}\n{json.dumps(booking_ctx, default=str)}"
    )


def _get_retrieved_docs_msg(retrieved_docs: str) -> SystemMessage | None:
    if not retrieved_docs:
        return None
    return SystemMessage(
        content=f"{PromptLabels.RETRIEVED_DOCUMENTS}\n{retrieved_docs}"
    )


def _get_search_msg(intent: str, search_results: list[dict]) -> SystemMessage | None:
    if not search_results:
        if intent == "search":
            return SystemMessage(content="No results found.")
        return None

    label = "" if intent == "search" else f"{PromptLabels.PROPERTY_SEARCH_RESULTS}\n"
    return SystemMessage(content=f"{label}{format_listings(search_results)}")


def _get_plain_llm():
    return get_llm()


def _get_tool_llm():
    return get_llm().bind_tools(get_all_tools())


def _needs_tools(state: RealEstateAgentState) -> bool:
    """True on the first (tool-calling) pass only: action intent with HumanMessage last."""
    intent = state.get(StateKeys.USER_INTENT, "general")
    if intent not in IntentConfig.TOOL_INTENTS:
        return False

    if not state["messages"]:
        return False

    return isinstance(state["messages"][-1], HumanMessage)
