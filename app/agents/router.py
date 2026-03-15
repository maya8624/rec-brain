# app/agents/router.py
"""
Conditional edge functions for the LangGraph graph.

Router map:
    route_agent_output    → after agent_node
                            "vector_search" | "sql_search" | "tools" | "end"

    route_after_search    → after vector_search_node or sql_search_node
                            "agent" | "end"

    route_after_tools     → after tool_node
                            "context_update" | "safety" | "end"

    route_after_context   → after context_update_node
                            "agent" | "end"

    route_after_safety    → after safety_node
                            "agent" | "end"

Design principle:
    Routers are PURE functions — they only read state, never modify it.
    All state changes happen inside nodes.
"""
import json
import logging

from langchain_core.messages import AIMessage, ToolMessage

from app.agents.state import RealEstateAgentState

logger = logging.getLogger(__name__)

# ── Tool sets ──────────────────────────────────────────────────────────────────
SQL_TOOLS = {"search_listings", "check_availability"}
VECTOR_TOOLS = {"search_documents"}
ACTION_TOOLS = {"book_inspection", "cancel_inspection"}


# ── Router: after agent_node ───────────────────────────────────────────────────

def route_agent_output(state: RealEstateAgentState) -> str:
    """
    Core routing decision after the LLM responds.

    Reads tool_calls from the last AIMessage and routes to:
        "vector_search" — LLM called search_documents
        "sql_search"    — LLM called search_listings / check_availability
        "tools"         — LLM called book_inspection / cancel_inspection
        "end"           — plain text response or escalation
    """
    if state.get("requires_human"):
        logger.info("route_agent_output | escalation → end")
        return "end"

    last_message = _last_ai_message(state)

    if last_message is None:
        logger.warning("route_agent_output | no AI message → end")
        return "end"

    if not getattr(last_message, "tool_calls", None):
        logger.info("route_agent_output | plain response → end")
        return "end"

    tool_names = {tc["name"] for tc in last_message.tool_calls}

    has_vector = bool(tool_names & VECTOR_TOOLS)
    has_sql = bool(tool_names & SQL_TOOLS)
    has_action = bool(tool_names & ACTION_TOOLS)

    if has_vector:
        logger.info("route_agent_output | tools=%s → vector_search", tool_names)
        return "vector_search"

    if has_sql:
        logger.info("route_agent_output | tools=%s → sql_search", tool_names)
        return "sql_search"

    if has_action:
        logger.info("route_agent_output | tools=%s → tools", tool_names)
        return "tools"

    logger.warning(
        "route_agent_output | unrecognised tools=%s → end", tool_names)
    return "end"


# ── Router: after vector_search_node or sql_search_node ───────────────────────

def route_after_search(state: RealEstateAgentState) -> str:
    """
    Always returns to agent so LLM can formulate a response
    from the search results.
    """
    if state.get("requires_human"):
        logger.info("route_after_search | escalation → end")
        return "end"

    logger.info("route_after_search | → agent")
    return "agent"


# ── Router: after tool_node ────────────────────────────────────────────────────

def route_after_tools(state: RealEstateAgentState) -> str:
    """
    → "context_update"  at least one tool succeeded
    → "safety"          all tools failed
    → "end"             escalation flag set
    """
    if state.get("requires_human"):
        logger.info("route_after_tools | escalation → end")
        return "end"

    tool_results = _extract_tool_results(list(state["messages"]))

    if not tool_results:
        logger.warning("route_after_tools | no tool results → safety")
        return "safety"

    if all(not r.get("success", False) for r in tool_results):
        logger.warning(
            "route_after_tools | all %d tools failed → safety",
            len(tool_results),
        )
        return "safety"

    logger.info(
        "route_after_tools | %d/%d succeeded → context_update",
        sum(1 for r in tool_results if r.get("success")),
        len(tool_results),
    )
    return "context_update"


# ── Router: after context_update_node ─────────────────────────────────────────

def route_after_context(state: RealEstateAgentState) -> str:
    if state.get("requires_human"):
        logger.info("route_after_context | escalation → end")
        return "end"

    logger.info("route_after_context | → agent")
    return "agent"


# ── Router: after safety_node ─────────────────────────────────────────────────

def route_after_safety(state: RealEstateAgentState) -> str:
    if state.get("requires_human"):
        logger.warning("route_after_safety | threshold reached → end")
        return "end"

    logger.info("route_after_safety | under threshold → agent")
    return "agent"


# ── Private helpers ────────────────────────────────────────────────────────────

def _last_ai_message(state: RealEstateAgentState) -> AIMessage | None:
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            return msg
    return None


def _extract_tool_results(messages: list) -> list[dict]:
    """
    Parse the most recent batch of ToolMessages.
    Stops at the first AIMessage — only reads the latest batch.
    """
    results = []
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            break
        if isinstance(msg, ToolMessage):
            try:
                content = (
                    json.loads(msg.content)
                    if isinstance(msg.content, str)
                    else msg.content
                )
                results.append(
                    content if isinstance(content, dict)
                    else {"output": content}
                )
            except (json.JSONDecodeError, TypeError):
                results.append({"success": False, "error": str(msg.content)})
    return results
