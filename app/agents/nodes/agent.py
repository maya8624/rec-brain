"""
agent_node  — the primary LLM decision node.

The LLM receives the full conversation history plus a prepended system
prompt and decides whether to call tools or respond directly.

Intent classification runs on every turn via _classify_intent().
It is a zero-LLM-cost keyword heuristic — a hint for logging and
future fast-path routing, NOT a hard constraint on tool selection.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.state import RealEstateAgentState, UserIntent
from app.infrastructure.llm import get_llm
from app.prompts.agent import REAL_ESTATE_AGENT_SYSTEM
from app.tools import get_all_tools

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent keyword table
# Order matters — more specific intents listed first.
# "cancellation" before "booking" because "cancel my booking" contains
# booking keywords too; most-specific wins.
# ---------------------------------------------------------------------------
_INTENT_KEYWORDS: list[tuple[UserIntent, list[str]]] = [
    ("cancellation", [
        "cancel", "cancellation", "cancelled", "withdraw",
        "no longer", "don't want", "remove booking",
    ]),
    ("booking", [
        "book", "inspect", "inspection", "viewing", "view",
        "schedule", "arrange", "available", "availability",
        "when can i", "open for inspection", "open home",
    ]),
    ("document_query", [
        "lease", "contract", "strata", "terms", "clause",
        "bond", "deposit", "condition", "by-law", "bylaw",
        "pet policy", "break lease", "notice period",
        "landlord", "tenant", "agreement",
    ]),
    ("search", [
        "find", "search", "show", "list", "looking for",
        "properties", "house", "apartment", "unit", "townhouse",
        "bedroom", "bathroom", "suburb", "price", "budget",
        "under", "rent", "buy", "purchase",
    ]),
]


def _classify_intent(message: str) -> UserIntent:
    """
    Fast keyword-based intent classification. Zero LLM cost.

    Returns the first matching intent in priority order, or "general"
    for greetings, small talk, or anything unrecognised.
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


async def agent_node(state: RealEstateAgentState) -> dict:
    """
    Primary decision node — the LLM brain of the agent.

    Receives the full conversation history and decides one of:
        A) Call one or more tools  -> LangGraph routes to tool node
        B) Respond directly        -> LangGraph routes to END

    The system prompt is prepended each turn rather than stored in state
    to avoid bloating the checkpointer database.

    Returns partial state: { messages, user_intent }
    """
    llm = get_llm()
    tools = get_all_tools()
    llm_with_tools = llm.bind_tools(tools)

    last_human_msg = _get_last_human_message(state)
    intent = _classify_intent(last_human_msg)

    messages = [SystemMessage(
        content=REAL_ESTATE_AGENT_SYSTEM)] + list(state["messages"])

    logger.info(
        "agent_node | intent=%s | messages=%d | errors=%d",
        intent,
        len(state["messages"]),
        state.get("error_count", 0),
    )

    response = await llm_with_tools.ainvoke(messages)

    if hasattr(response, "tool_calls") and response.tool_calls:
        logger.info("agent_node | tool_calls=%s", [
                    tc["name"] for tc in response.tool_calls])

    return {
        "messages": [response],
        "user_intent": intent,
    }
