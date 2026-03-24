"""
Compiles the LangGraph StateGraph.

Graph topology:

    START
      │
      ▼
    intent_node
      │
      ├──► listing_search_node ──┐
      ├──► vector_search_node ───┼──► agent_node (format) ──► END
      │                          │
      ├──► agent_node ───────────┘
      │    (booking/cancellation)
      │         │
      │    tools_node
      │         │
      │    context_update ──► agent_node (format) ──► END
      │         │
      │       safety ──► agent_node or END
      │
      └──► END (early_response — compound intent)

TODO:
    - human_escalation_node: add AIMessage before END when requires_human=True
    - multi-step tool calling for compound intents
TODO:
- [ ] human_escalation_node — add AIMessage before END when requires_human=True
- [ ] Agency info storage — office hours, processes, policies
      Options: vector store (document_query intent) or dedicated DB node
      Leaning towards vector store — add keywords to document_query intent
"""
import logging
import os

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.core.constants import Node
from app.agents.nodes.agent import agent_node
from app.agents.nodes.context import context_update_node
from app.agents.nodes.intent import intent_node
from app.agents.nodes.listing_search import listing_search_node
from app.agents.nodes.safety import safety_node
from app.agents.nodes.vector import vector_search_node
from app.agents.router import (
    route_intent_output,
    route_agent_output,
    route_after_context,
    route_after_safety,
    route_after_search,
    route_after_tools,
)
from app.agents.state import RealEstateAgentState
from app.tools import get_all_tools

logger = logging.getLogger(__name__)


def build_graph():
    tools = get_all_tools()
    tool_node = ToolNode(tools)
    graph = StateGraph(RealEstateAgentState)

    # ------------------------
    # Nodes
    # ------------------------
    graph.add_node(Node.INTENT,          intent_node)
    graph.add_node(Node.LISTING_SEARCH,  listing_search_node)
    graph.add_node(Node.VECTOR_SEARCH,   vector_search_node)
    graph.add_node(Node.AGENT,           agent_node)
    graph.add_node(Node.TOOLS,           tool_node)
    graph.add_node(Node.CONTEXT_UPDATE,  context_update_node)
    graph.add_node(Node.SAFETY,          safety_node)

    # ------------------------
    # Entry point
    # ------------------------
    graph.set_entry_point(Node.INTENT)

    # ------------------------
    # Intent routing
    # ------------------------
    graph.add_conditional_edges(
        source=Node.INTENT,
        path=route_intent_output,
        path_map={
            Node.LISTING_SEARCH: Node.LISTING_SEARCH,
            Node.VECTOR_SEARCH:  Node.VECTOR_SEARCH,
            Node.AGENT:          Node.AGENT,
            Node.END:            END,
        },
    )

    # ------------------------
    # listing_search → agent (format)
    # vector_search  → agent (format)
    # ------------------------
    graph.add_conditional_edges(
        source=Node.LISTING_SEARCH,
        path=route_after_search,
        path_map={Node.AGENT: Node.AGENT, Node.END: END},
    )

    graph.add_conditional_edges(
        source=Node.VECTOR_SEARCH,
        path=route_after_search,
        path_map={Node.AGENT: Node.AGENT, Node.END: END},
    )

    # ------------------------
    # Agent output routing
    # booking/cancellation → tools
    # general/format pass  → END
    # ------------------------
    graph.add_conditional_edges(
        source=Node.AGENT,
        path=route_agent_output,
        path_map={
            Node.TOOLS: Node.TOOLS,
            Node.END:   END,
        },
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
    # Context update → agent (format)
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
