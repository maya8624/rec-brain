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

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.core.constants import Node
from app.agents.nodes.agent import agent_node
from app.agents.nodes.context import context_update_node
from app.agents.nodes.safety import safety_node
from app.agents.nodes.sql import sql_search_node
from app.agents.nodes.vector import vector_search_node
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

    # ------------------------
    # Nodes
    # ------------------------
    graph.add_node(Node.AGENT,          agent_node)
    graph.add_node(Node.VECTOR_SEARCH,  vector_search_node)
    graph.add_node(Node.SQL_SEARCH,     sql_search_node)
    graph.add_node(Node.TOOLS,          tool_node)
    graph.add_node(Node.CONTEXT_UPDATE, context_update_node)
    graph.add_node(Node.SAFETY,         safety_node)

    graph.set_entry_point(Node.AGENT)

    # ------------------------
    # Agent output routing — fan out to 3 paths
    # ------------------------
    graph.add_conditional_edges(
        source=Node.AGENT,
        path=route_agent_output,
        path_map={
            Node.VECTOR_SEARCH: Node.VECTOR_SEARCH,
            Node.SQL_SEARCH:    Node.SQL_SEARCH,
            Node.TOOLS:         Node.TOOLS,
            Node.END:           END,
        },
    )

    # ------------------------
    # Search paths — both go back to agent
    # ------------------------
    graph.add_conditional_edges(
        source=Node.VECTOR_SEARCH,
        path=route_after_search,
        path_map={Node.AGENT: Node.AGENT, Node.END: END},
    )

    graph.add_conditional_edges(
        source=Node.SQL_SEARCH,
        path=route_after_search,
        path_map={Node.AGENT: Node.AGENT, Node.END: END},
    )

    # ------------------------
    # Action tools
    # ------------------------
    graph.add_conditional_edges(
        source=Node.TOOLS,
        path=route_after_tools,
        path_map={
            Node.CONTEXT_UPDATE: Node.CONTEXT_UPDATE,
            Node.SAFETY:         Node.SAFETY,
            Node.END:            END,
        },
    )

    # ------------------------
    # Context update → agent
    # ------------------------
    graph.add_conditional_edges(
        source=Node.CONTEXT_UPDATE,
        path=route_after_context,
        path_map={Node.AGENT: Node.AGENT, Node.END: END},
    )

    # ------------------------
    # Safety → agent or end
    # ------------------------
    graph.add_conditional_edges(
        source=Node.SAFETY,
        path=route_after_safety,
        path_map={Node.AGENT: Node.AGENT, Node.END: END},
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

    logger.info("Using InMemorySaver (development)")
    return InMemorySaver()


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
        return InMemorySaver()

    except Exception as e:
        logger.error(
            "PostgresSaver failed: %s — falling back to MemorySaver", e)
        return InMemorySaver()
