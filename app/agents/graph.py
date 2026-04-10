"""
Compiles the LangGraph StateGraph.

Graph topology:

    START
      │
      ▼
    intent_node
      │
      ├──► listing_search_node ──┐
      ├──► vector_search_node ───┤
      ├──► hybrid_search_node ───┼──► agent_node (format) ──► END
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
- [ ] human_escalation_node — add AIMessage before END when requires_human=True
- [ ] multi-step tool calling for compound intents
- [ ] Agency info storage — office hours, processes, policies
      Options: vector store (document_query intent) or dedicated DB node
      Leaning towards vector store — add keywords to document_query intent
"""
import logging

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from app.core.constants import Node
from app.agents.nodes.agent import agent_node
from app.agents.nodes.context import context_update_node
from app.agents.nodes.hybrid import hybrid_search_node
from app.agents.nodes.intent import intent_node
from app.agents.nodes.listing import listing_search_node
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


def build_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    '''
    Constructs the StateGraph with nodes and edges.
    Called once at app startup to create the compiled graph.
    '''
    tools = get_all_tools()
    tool_node = ToolNode(tools)

    # ------------------------
    # Create a graph with RealEstateAgentState as the state type
    # ------------------------
    graph = StateGraph(RealEstateAgentState)

    # ------------------------
    # Nodes
    # ------------------------
    graph.add_node(Node.INTENT,          intent_node)
    graph.add_node(Node.LISTING_SEARCH,  listing_search_node)
    graph.add_node(Node.VECTOR_SEARCH,   vector_search_node)
    graph.add_node(Node.HYBRID_SEARCH,   hybrid_search_node)
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
            Node.HYBRID_SEARCH:  Node.HYBRID_SEARCH,
            Node.AGENT:          Node.AGENT,
            Node.END:            END,
        },
    )

    # ------------------------
    # listing_search → agent (format)
    # vector_search  → agent (format)
    # hybrid_search  → agent (format)
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

    graph.add_conditional_edges(
        source=Node.HYBRID_SEARCH,
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

    compiled = graph.compile(checkpointer=checkpointer)

    logger.info(
        "LangGraph compiled | nodes=%s | checkpointer=%s",
        list(graph.nodes),
        type(checkpointer).__name__,
    )

    return compiled
