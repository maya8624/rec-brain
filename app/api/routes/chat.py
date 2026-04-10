"""
All traffic arrives from .NET backend — never directly from React.
thread_id(session_id) from .NET maps to LangGraph thread_id for state persistence.

No AI logic here — only HTTP concerns:
    parsing, routing to agent, formatting response, error handling.
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents.state import initial_state
from app.api.dependencies import get_agent
from app.schemas.chat import ChatErrorResponse, ChatRequest, ChatResponse, SourceDocument

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
        request: ChatRequest,
        http_request: Request,
        agent=Depends(get_agent)) -> ChatResponse:
    """
    Main chat endpoint — called by .NET backend for every user message.
    """

    logger.info("chat | thread_id=%s | user=%s | new=%s",
                request.thread_id, request.user_id, request.is_new_conversation)

    try:
        config = {
            "configurable": {
                "thread_id": request.thread_id,
                "request": http_request,
            }
        }

        if request.is_new_conversation:
            input_state = initial_state()
            input_state["messages"] = [HumanMessage(content=request.message)]

        else:
            # LangGraph rehydrates existing state from checkpointer automatically
            input_state = {"messages": [HumanMessage(content=request.message)]}

        result = await agent.ainvoke(input_state, config=config)

        return _build_response(request.thread_id, result)

    except Exception as exc:
        logger.exception("chat | error | thread_id=%s | %s",
                         request.thread_id, exc)

        raise HTTPException(
            status_code=500,
            detail=ChatErrorResponse(
                error="AI service error. Please try again.",
                thread_id=request.thread_id,
            ).model_dump(),  # convert Pydantic model to dict for JSON response
        ) from exc


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    http_request: Request,
    agent=Depends(get_agent),
) -> StreamingResponse:
    """
    Streaming chat — returns tokens via Server-Sent Events.
    Connect from React using EventSource or fetch with streaming.

    SSE events:
        data: {"type": "token", "content": "Hello"}
        data: {"type": "tool_start", "tool": "search_listings"}
        data: {"type": "tool_end", "tool": "search_listings"}
        data: {"type": "error", "message": "..."}
        data: [DONE]
    """
    request_id = getattr(http_request.state, "request_id", "unknown")

    logger.info(
        "chat/stream | thread_id=%s | user=%s | id=%s",
        request.thread_id, request.user_id, request_id,
    )

    return StreamingResponse(
        _event_generator(request, http_request, agent),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",      # disable nginx buffering
            "X-Request-ID": request_id,
        },
    )


async def _event_generator(request: ChatRequest, http_request: Request, agent):
    """
    Mirrors the pattern from your original stream_agent() approach.
    """
    try:
        config = {
            "configurable": {
                "thread_id": request.thread_id,
                "request": http_request,
            }
        }

        if request.is_new_conversation:
            input_state = initial_state()
            input_state["messages"] = [HumanMessage(content=request.message)]
        else:
            input_state = {"messages": [HumanMessage(content=request.message)]}

        emitted_tokens = False
        async for event in agent.astream_events(input_state, config=config, version="v2"):
            # Capture early_response from graph end event (compound intent)
            if event.get("event") == "on_chain_end" and event.get("name") == "LangGraph":
                output = event.get("data", {}).get("output", {})
                early = output.get("early_response")
                if early and not emitted_tokens:
                    yield f"data: {json.dumps({'type': 'token', 'content': early})}\n\n"
                continue

            sse = _to_sse_event(event)
            if sse:
                if sse.get("type") == "token":
                    emitted_tokens = True
                yield f"data: {json.dumps(sse)}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as exc:
        logger.exception(
            "chat/stream | error | thread_id=%s | %s",
            request.thread_id, exc,
        )
        yield f"data: {json.dumps({'type': 'error', 'message': 'An unexpected error occurred.'})}\n\n"


def _to_sse_event(event: dict) -> dict | None:
    """
    Converts a LangGraph astream_events dict into an SSE payload.
    Returns None for internal events we don't surface to the frontend.
    """
    kind = event.get("event", "")
    name = event.get("name", "")

    # LLM is generating a token — send it immediately
    if kind == "on_chat_model_stream":
        chunk = event.get("data", {}).get("chunk")
        if chunk and hasattr(chunk, "content") and chunk.content:
            return {"type": "token", "content": chunk.content}

    # Tool starting — frontend can show "Searching..." indicator
    if kind == "on_tool_start":
        return {"type": "tool_start", "tool": name}

    # Tool finished — frontend can hide indicator
    if kind == "on_tool_end":
        return {"type": "tool_end", "tool": name}

    return None


# ── Response builder ───────────────────────────────────────────────────────────

def _build_response(thread_id: str, result: dict) -> ChatResponse:
    """
    Builds a ChatResponse from the final LangGraph state dict.
    Extracts reply text, tools used, booking state, RAG sources.
    """

    ai_messages = [
        message for message in result.get("messages", [])
        if isinstance(message, AIMessage)
    ]

    reply = (
        ai_messages[-1].content
        if ai_messages
        else result.get("early_response") or "I couldn't process that request."
    )

    booking_status = result.get("booking_status", {})
    booking_context = result.get("booking_context", {})

    return ChatResponse(
        reply=reply,
        thread_id=thread_id,
        tools_used=_extract_tools_used(result.get("messages", [])),
        intent=result.get("user_intent", "unknown"),
        booking_confirmed=booking_status.get("confirmed", False),
        booking_cancelled=booking_status.get("cancelled", False),
        confirmation_id=booking_context.get("confirmation_id"),
        requires_human=result.get("requires_human", False),
        sources=_extract_sources(result.get("messages", [])),
    )


def _extract_tools_used(messages: list) -> list[str]:
    """Collect unique tool names the agent called this turn."""

    tools: list[str] = []

    for msg in messages:
        if isinstance(msg, AIMessage):
            for tool_call in getattr(msg, "tool_calls", []):
                name = tool_call.get("name", "")
                if name and name not in tools:
                    tools.append(name)
    return tools


def _extract_sources(messages: list) -> list[SourceDocument]:
    """Extract RAG source documents from search_documents tool result."""

    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name == "search_documents":
            try:
                content = (
                    json.loads(message.content)
                    if isinstance(message.content, str)
                    else message.content
                )

                return [
                    SourceDocument(
                        document=source.get("document", ""),
                        doc_type=source.get("doc_type", ""),
                        page=str(source.get("page", "")),
                        relevance_score=float(
                            source.get("relevance_score", 0)),
                    )

                    for source in content.get("sources", [])
                ]
            except (json.JSONDecodeError, TypeError, KeyError):
                pass
    return []
