"""
agent_node              — the primary LLM decision node.
_classify_intent        — zero-cost keyword intent heuristic
_get_last_human_message — extracts latest user message from state
"""

import logging
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.state import RealEstateAgentState, UserIntent
from app.infrastructure.llm import get_llm
from app.prompts.agent import REAL_ESTATE_AGENT_SYSTEM
from app.tools import get_all_tools

logger = logging.getLogger(__name__)

_LLM_WITH_TOOLS = get_llm().bind_tools(get_all_tools())

# ---------------------------------------------------------------------------
# Intent keyword table
# Order matters — more specific intents listed first.
# "cancellation" before "booking" because "cancel my booking" contains
# booking keywords too; most-specific wins.
# ---------------------------------------------------------------------------
_INTENT_KEYWORDS: list[tuple[UserIntent, frozenset[str]]] = [
    ("cancellation", frozenset([
        "cancel", "cancellation", "cancelled", "withdraw",
        "no longer", "don't want", "remove booking",
    ])),
    ("booking", frozenset([
        "book", "inspect", "inspection", "viewing", "view",
        "schedule", "arrange", "available", "availability",
        "when can i", "open for inspection", "open home",
    ])),
    ("document_query", frozenset([
        "lease", "contract", "strata", "terms", "clause",
        "bond", "deposit", "condition", "by-law", "bylaw",
        "pet policy", "break lease", "notice period",
        "landlord", "tenant", "agreement",
    ])),
    ("search", frozenset([
        "find", "search", "show", "list", "looking for",
        "properties", "house", "apartment", "unit", "townhouse",
        "bedroom", "bathroom", "suburb", "price", "budget",
        "under", "rent", "buy", "purchase",
    ])),
]


async def agent_node(state: RealEstateAgentState) -> dict[str, Any]:
    """
    Primary decision node — the LLM brain of the agent.

    Receives the full conversation history and decides one of:
        A) Call one or more tools  -> LangGraph routes to tool node
        B) Respond directly        -> LangGraph routes to END

    The system prompt is prepended each turn rather than stored in state
    to avoid bloating the checkpointer database.
    """

    last_human_msg = _get_last_human_message(state)
    intent = _classify_intent(last_human_msg)

    messages = [
        SystemMessage(content=REAL_ESTATE_AGENT_SYSTEM), *
        state["messages"]  # Unpacking the list of messages from state
    ]

    logger.info(
        "agent_node | intent=%s | messages=%d | errors=%d",
        intent,
        len(state["messages"]),
        state.get("error_count", 0),
    )

    response = await _LLM_WITH_TOOLS.ainvoke(messages)

    if hasattr(response, "tool_calls") and response.tool_calls:
        logger.info("agent_node | tool_calls=%s", [
                    tc["name"] for tc in response.tool_calls])

    return {
        "messages": [response],
        "user_intent": intent,
    }


def _classify_intent(message: str) -> UserIntent:
    """
    Fast keyword-based intent classification. Zero LLM cost.
    Returns the first matching intent in priority order, or "general"
    for greetings, small talk, or anything unrecognised.

    Note: uses substring matching — "view" matches "overview".
    Acceptable for this domain; use word-boundary matching if false
    positives become a problem.
    """
    if not message:
        return "general"

    msg_lower = message.lower()

    for intent, keywords in _INTENT_KEYWORDS:
        if any(keyword in msg_lower for keyword in keywords):
            logger.debug("_classify_intent | '%.40s' -> %s", message, intent)
            return intent

    return "general"


def _get_last_human_message(state: RealEstateAgentState) -> str:
    """Return the content of the most recent HumanMessage in state."""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.content if isinstance(message.content, str) else ""

    return ""
