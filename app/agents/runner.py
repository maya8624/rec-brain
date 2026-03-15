"""
Agent runner — handles invoking the agent and streaming responses.
Separated from builder.py: building is a one-time setup, running is per-request.
"""

from typing import AsyncGenerator
import structlog
from langgraph.graph.state import CompiledStateGraph

from app.models.state_models import OrchestrationState

logger = structlog.get_logger(__name__)


async def run_agent(agent: CompiledStateGraph, state: OrchestrationState) -> OrchestrationState:
    """
    Invoke the agent for a single turn and return the final text response.
    session_id maps to a conversation thread — same ID = shared memory/history.
    """
    log = logger.bind(session_id=state.session_id)
    config = {"configurable": {"thread_id": state.session_id}}

    try:
        result = await agent.ainvoke(
            {"messages": [("user", state.user_message)]},
            config=config,
        )

        # LangGraph expects messages in this format.
        # It's a list of tuples where the first item is the role (`"user"`, `"assistant"`)
        # and the second is the content.
        # result["messages"] is the full conversation list after the agent finishes —
        # including all tool calls and intermediate steps.
        # `[-1]` gets the **last** message which is always the agent's final response to the user. `.content` extracts the text.
        #  result["messages"] = [
        # HumanMessage("I want to book 12 Ocean Drive..."),
        # AIMessage("Let me check availability..."),      # agent thinking
        # ToolMessage(check_availability result),          # tool result
        # AIMessage("✅ Appointment confirmed!...")        # ← [-1] this one
        response = result["messages"][-1].content
        log.info("Agent responded", response_length=len(response))

        return OrchestrationState(
            session_id=state.session_id,
            user_id=state.user_id,
            user_message=state.user_message,
            answer=response
        )

    except Exception as e:
        log.error("Agent invocation failed", error=str(e))
        raise


async def stream_agent(agent: CompiledStateGraph,
                       user_message: str,
                       session_id: str) -> AsyncGenerator[str, None]:
    """
    Stream the agent's response token by token.
    Used by the SSE endpoint for real-time output to the React frontend.
    """
    log = logger.bind(session_id=session_id)
    config = {"configurable": {"thread_id": session_id}}

    try:
        async for event in agent.astream_events(
            {"messages": [("user", user_message)]},
            config=config,
            version="v2",
        ):
            kind = event.get("event")
            # Stream only the final LLM text tokens
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    yield chunk.content

    except Exception as e:
        log.error("Agent streaming failed", error=str(e))
        yield "⚠️ Something went wrong. Please try again."

# `astream_events()` fires many events as the agent runs — tool calls starting, tool calls finishing,
# LLM tokens arriving etc. We only care about `"on_chat_model_stream"` which is the event that fires for each new token from Groq.
# ```
# Events fired during one agent run:
#   on_chain_start          ← agent loop starts
#   on_chat_model_start     ← LLM starts thinking
#   on_chat_model_stream    ← token: "✅"      ← we yield this
#   on_chat_model_stream    ← token: " Appoint" ← we yield this
#   on_chat_model_stream    ← token: "ment"    ← we yield this
#   on_tool_start           ← tool call starts
#   on_tool_end             ← tool call ends
#   on_chat_model_stream    ← token: " confirmed" ← we yield this
#   on_chain_end            ← agent loop ends
# ```

# Each yielded token flows back up to `chat.py`'s `event_generator()` which wraps it in SSE format and sends it to React.

# ---
