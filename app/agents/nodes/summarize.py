"""
summarize_node — builds a rolling summary of conversation turns that have
fallen outside the active history window.

Runs after agent_node completes a non-tool intent turn. The summary covers
everything older than the intent's history window so the LLM and RAG retriever
retain long-range context without expanding the token budget.

The update is incremental: only newly overflowed turns (since the last summary
run) are fed to the LLM together with the existing summary, so cost stays O(delta).
"""

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.state import RealEstateAgentState
from app.core.constants import Intent, IntentConfig, StateKeys
from app.infrastructure.llm import get_llm
from app.prompts.summarize import CONVERSATION_SUMMARIZE_PROMPT

logger = structlog.get_logger(__name__)


async def summarize_node(state: RealEstateAgentState) -> dict:
    intent = state.get(StateKeys.USER_INTENT, Intent.GENERAL)
    if intent not in IntentConfig.SUMMARY_INTENTS:
        return {}

    all_msgs = [
        message for message in state["messages"]
        if isinstance(message, (HumanMessage, AIMessage))
    ]
    window = IntentConfig.HISTORY_BY_INTENT.get(intent, IntentConfig.DEFAULT_HISTORY_LIMIT)

    if len(all_msgs) <= window:
        return {}

    overflow = all_msgs[:-window]
    current_property_id = (state.get("property_context") or {}).get("property_id")
    existing = state.get(StateKeys.CONVERSATION_SUMMARY)
    already_summarised = state.get(StateKeys.SUMMARY_MESSAGE_COUNT) or 0

    # Evict stale summary when the user switches to a different property.
    # Only evict when the summary was built for a specific property that no longer matches —
    # if summary_property_id is None the summary was built without a property focus and
    # remains valid as general context.
    summary_property_id = state.get(StateKeys.SUMMARY_PROPERTY_ID)
    if (
        existing
        and summary_property_id is not None
        and current_property_id != summary_property_id
    ):
        logger.info(
            "summarize_node_evicted",
            previous_property_id=summary_property_id,
            current_property_id=current_property_id,
        )
        existing = None
        already_summarised = 0

    if len(overflow) <= already_summarised:
        return {}

    newly_overflowed = overflow[already_summarised:]

    prompt: list = [SystemMessage(content=CONVERSATION_SUMMARIZE_PROMPT)]
    if existing:
        prompt.append(SystemMessage(content=f"Existing summary:\n{existing}"))
    prompt.extend(newly_overflowed)

    try:
        response: AIMessage = await get_llm().ainvoke(prompt)
        new_summary = response.content.strip()
        logger.info(
            "summarize_node_updated",
            delta_messages=len(newly_overflowed),
            summary_length=len(new_summary),
            property_id=current_property_id,
        )
        return {
            StateKeys.CONVERSATION_SUMMARY: new_summary,
            StateKeys.SUMMARY_MESSAGE_COUNT: len(overflow),
            StateKeys.SUMMARY_PROPERTY_ID: current_property_id,
        }
    except Exception as exc:
        logger.warning("summarize_node_failed", error=str(exc))
        return {}
