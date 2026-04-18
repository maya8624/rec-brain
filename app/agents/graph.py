"""
Compiles the LangGraph StateGraph.

Graph topology:

    START
      │
      ▼
    intent_node  (hybrid classifier — keyword fast path + LLM structured output)
      │            LLM path extracts entities → state["search_context"]
      │            (location, property_type, bedrooms, max_price, listing_type, etc.)
      │
      ├──► listing_search_node ──┐
      ├──► vector_search_node ───┤
      ├──► hybrid_search_node ───┼──► agent_node (format reply) ──► END
      │                          │
      ├──► agent_node ───────────┘
      │    (booking / cancellation — tool-calling mode)
      │         │
      │    tools_node  (check_availability / book / cancel)
      │         │
      │    context_update_node  (writes booking_context / booking_status)
      │         │
      │    safety_node  (error counting → requires_human flag)
      │         │
      │    agent_node (format reply) ──► END
      │
      └──► END (early_response — vague query or compound intent, user must clarify)

intent_node strategy (see app/agents/nodes/intent.py):
  - Fast path (no LLM): obvious cancellation and standalone booking via keywords
  - LLM path: everything else — search, follow-ups, compound intents, general
    Uses last 4 messages as history to resolve follow-up references
    Returns IntentClassification (Pydantic) with intent + extracted search entities

search results (see app/agents/nodes/agent.py):
  - Search nodes write results to state["retrieved_docs"] (plain assignment)
    instead of state["messages"] — prevents stale results accumulating in history
  - agent_node injects retrieved_docs as an ephemeral SystemMessage at the end
    of the prompt, then clears it to None after each LLM call
  - History depth is intent-aware:
      booking / cancellation  → 10 turns  (multi-turn contact collection)
      search / hybrid_search  →  6 turns
      document_query / general →  4 turns

TODO:
- [ ] human_escalation_node — requires_human=True currently exits via every
      router's _requires_human() guard with no AIMessage. The escalation reply
      is handled for now in _build_response (app/api/routes/chat.py) as a
      formatting fallback. Promote to a dedicated graph node when escalation
      needs side effects: webhook calls, CRM notifications, staff alerts, etc.
      Implementation: insert human_escalation_node between safety_node (or any
      _requires_human guard) and END; it appends a canned AIMessage and can
      trigger external calls. All _requires_human guards in router.py route to
      it instead of END.
- [ ] multi-step tool calling for compound intents
- [ ] Use state["search_context"] entities in listing_search_node to build SQL
      from structured data instead of re-parsing the raw message via sql_service
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
