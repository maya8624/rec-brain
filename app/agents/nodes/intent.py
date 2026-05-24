"""
intent_node — classifies user intent before routing through the graph.

Strategy: hybrid keyword + LLM

    Fast path (keyword, no LLM):
        - Obvious cancellation      → "cancellation"
        - Obvious booking lookup    → "booking_lookup"
        - Obvious booking only      → "booking"
        - Deposit + search keywords → "search_then_deposit"
        - Deposit only              → "deposit_payment"
        - Slot selection in context → "booking" (continuation)

    LLM path (everything else):
        - Ambiguous, follow-up, compound, search queries
        - Returns IntentClassification with intent + extracted entities
        - Entities written to state["search_context"] for downstream nodes
"""

import logging
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.agents.nodes._base import last_human_message
from app.agents.nodes._fast_path import (
    is_booking_continuation,
    is_cancellation_continuation,
    fast_path_intent,
)
from app.agents.state import ConversationPhase, IntentClassification, RealEstateAgentState
from app.core.constants import AppStateKeys, Intent, IntentConfig, StateKeys
from app.infrastructure.llm import get_llm
from app.prompts.intent import INTENT_CLASSIFICATION_PROMPT

logger = logging.getLogger(__name__)


async def intent_node(state: RealEstateAgentState, config: RunnableConfig) -> dict[str, Any]:
    """
    Classifies intent from the latest HumanMessage.
    """
    forced = config.get(AppStateKeys.CONFIGURABLE, {}).get(AppStateKeys.FORCED_INTENT)
    if forced:
        return {StateKeys.USER_INTENT: forced}

    message = last_human_message(state).lower()
    if not message:
        return {StateKeys.USER_INTENT: Intent.GENERAL}

    obvious = fast_path_intent(message, state)
    if obvious:
        # elif obvious == "deposit_payment":
        #     obvious = _maybe_search_then_book(state)
        return {StateKeys.USER_INTENT: obvious}

    # TODO: should be in _fast_path
    if is_booking_continuation(state, message):
        return {StateKeys.USER_INTENT: Intent.BOOKING}

    # TODO: should be in _fast_path
    if is_cancellation_continuation(state, message):
        return {StateKeys.USER_INTENT: Intent.CANCELLATION}

    return await _classify_with_llm(state)


async def _classify_with_llm(state: RealEstateAgentState) -> dict[str, Any]:
    """
    Classifies intent via LLM when the fast path yields no result.

    Sends only HumanMessages (up to CLASSIFIER_HISTORY_LIMIT) so the LLM isn't
    distracted by tool results or agent reasoning from prior turns. If the previous
    intent completed, only the latest message is sent to treat the turn as fresh.

    Returns a typed IntentClassification object via with_structured_output — intent
    plus any extracted entities (location, bedrooms, price, etc.). Falls back to
    "general" on any LLM error to keep the conversation alive.
    """
    intent_completed = state.get(StateKeys.INTENT_COMPLETED, False)
    if intent_completed:
        history = state["messages"][-1:]
    else:
        human_messages = [
            msg for msg in state["messages"]
            if isinstance(msg, HumanMessage)
        ]
        history = human_messages[-IntentConfig.CLASSIFIER_HISTORY_LIMIT:]

    state_hint = _build_state_hint(state, intent_completed)

    content = INTENT_CLASSIFICATION_PROMPT + state_hint
    prompt = [SystemMessage(content=content), *history]

    try:
        llm = get_llm().with_structured_output(IntentClassification)
        classification: IntentClassification = await llm.ainvoke(prompt)
    except Exception as exc:
        logger.error("intent_node | LLM classification failed: %s", exc)
        return {StateKeys.USER_INTENT: Intent.GENERAL}

    return _build_intent_state_update(state, classification)


def _build_intent_state_update(
        state: RealEstateAgentState,
        classification: IntentClassification) -> dict[str, Any]:
    """
    Build the state update dict from an LLM classification result.

    Skips entity extraction for intents that don't drive SQL search ("general",
    "document_query"). For all others, replaces search_context on a new location
    or merges new entities into the existing context when location is unchanged.
    """
    update = {
        StateKeys.USER_INTENT:      classification.intent,
        StateKeys.EARLY_RESPONSE:   classification.early_response,
        StateKeys.INTENT_COMPLETED: False,
    }

    if classification.intent in (Intent.GENERAL, Intent.DOCUMENT_QUERY):
        return update

    entities = _extract_entities(classification)
    if not entities:
        return update

    existing = dict(state.get(StateKeys.SEARCH_CONTEXT) or {})
    update[StateKeys.SEARCH_CONTEXT] = (
        entities if entities.get("location")
        else {**existing, **entities}
    )

    return update


def _build_state_hint(state: RealEstateAgentState, intent_completed: bool) -> str:
    """Return a [STATE] hint string for the LLM prompt."""
    phase = state.get(StateKeys.PHASE, ConversationPhase.IDLE)
    if phase == ConversationPhase.IDLE and not intent_completed:
        return ""

    hint = ("\n[STATE]\n"
            f"phase: {phase.value}\n"
            f"completed: {intent_completed}\n"
            "rule: if completed=true, treat the latest user message as a fresh request.\n")

    return hint


def _extract_entities(classification: IntentClassification) -> dict:
    """Build a partial SearchContext dict from LLM-extracted entities."""
    _str_fields = ("location", "address", "listing_type", "property_type")
    _num_fields = ("bedrooms", "bathrooms", "max_price", "min_price", "limit")

    entities = {
        field: value
        for field in _str_fields
        if (value := getattr(classification, field))
    }
    entities |= {
        field: value
        for field in _num_fields
        if (value := getattr(classification, field)) is not None
    }

    if "limit" in entities:
        entities["limit"] = min(entities["limit"], 10)

    return entities
