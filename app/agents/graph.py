# app/agents/graph.py
"""
Compiles the LangGraph StateGraph.

Graph topology:

    START
      │
      ▼
    agent_node
      │
      ├──► vector_search_node ──┐
      ├──► sql_search_node ─────┤──► agent_node ──► END
      │                         │
      └──► tool_node ────────────┤
                │                │
          context_update ────────┘
                │
            safety ──► agent_node or END

Note:
    hybrid_node excluded for now — add once core flow is stable.
"""
import logging
import os

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.agents.nodes.nodes import agent_node, context_update_node, safety_node
from app.agents.nodes.search import sql_search_node, vector_search_node
from app.agents.router import (
    route_after_context,
    route_after_safety,
    route_after_search,
    route_after_tools,
    route_agent_output,
)
from app.agents.state import RealEstateAgentState
from app.tools import get_all_tools

logger = logging.getLogger(__name__)


def build_graph():
    """
    Builds and compiles the LangGraph agent.
    Called once at startup via get_agent().
    """
    tools = get_all_tools()
    tool_node = ToolNode(tools)

    graph = StateGraph(RealEstateAgentState)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    graph.add_node("agent",          agent_node)
    graph.add_node("vector_search",  vector_search_node)
    graph.add_node("sql_search",     sql_search_node)
    graph.add_node("tools",          tool_node)
    graph.add_node("context_update", context_update_node)
    graph.add_node("safety",         safety_node)

    # ── Entry ───────────────────────────────────────────────────────────────────
    graph.set_entry_point("agent")

    # ── After agent — fan out to 3 paths ───────────────────────────────────────
    graph.add_conditional_edges(
        "agent",
        route_agent_output,
        {
            "vector_search": "vector_search",
            "sql_search":    "sql_search",
            "tools":         "tools",
            "end":           END,
        },
    )

    # ── Search paths — both go back to agent ───────────────────────────────────
    graph.add_conditional_edges(
        "vector_search",
        route_after_search,
        {"agent": "agent", "end": END},
    )

    graph.add_conditional_edges(
        "sql_search",
        route_after_search,
        {"agent": "agent", "end": END},
    )

    # ── Action tools ────────────────────────────────────────────────────────────
    graph.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "context_update": "context_update",
            "safety":         "safety",
            "end":            END,
        },
    )

    # ── Context update → agent ──────────────────────────────────────────────────
    graph.add_conditional_edges(
        "context_update",
        route_after_context,
        {"agent": "agent", "end": END},
    )

    # ── Safety → agent or end ───────────────────────────────────────────────────
    graph.add_conditional_edges(
        "safety",
        route_after_safety,
        {"agent": "agent", "end": END},
    )

    checkpointer = _get_checkpointer()

    compiled = graph.compile(checkpointer=checkpointer)

    logger.info(
        "LangGraph compiled | nodes=%s | checkpointer=%s",
        list(graph.nodes),
        type(checkpointer).__name__,
    )

    return compiled


def _get_checkpointer():
    env = os.getenv("ENVIRONMENT", "development")

    if env == "production":
        return _build_postgres_checkpointer()

    logger.info("Using MemorySaver (development)")
    return MemorySaver()


def _build_postgres_checkpointer():
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from app.core.config import settings

        logger.info("Using PostgresSaver (production)")
        checkpointer = PostgresSaver.from_conn_string(settings.DATABASE_URL)
        checkpointer.setup()
        return checkpointer

    except ImportError:
        logger.error(
            "PostgresSaver not available — falling back to MemorySaver")
        return MemorySaver()

    except Exception as e:
        logger.error(
            "PostgresSaver failed: %s — falling back to MemorySaver", e)
        return MemorySaver()
