"""
Agent runner — handles invoking the agent and streaming responses.
Separated from builder.py: building is a one-time setup, running is per-request.
"""

import structlog
from typing import AsyncGenerator

logger = structlog.get_logger(__name__)


async def run_agent(agent, user_message: str, session_id: str) -> str:
    """
    Invoke the agent for a single turn and return the final text response.
    session_id maps to a conversation thread — same ID = shared memory/history.
    """
    log = logger.bind(session_id=session_id)
    config = {"configurable": {"thread_id": session_id}}

    try:
        result = await agent.ainvoke(
            {"messages": [("user", user_message)]},
            config=config,
        )
        response = result["messages"][-1].content
        log.info("Agent responded", response_length=len(response))
        return response

    except Exception as e:
        log.error("Agent invocation failed", error=str(e))
        raise


async def stream_agent(agent, user_message: str, session_id: str) -> AsyncGenerator[str, None]:
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
