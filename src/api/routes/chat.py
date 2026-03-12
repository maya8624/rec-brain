"""
Chat routes — POST /chat (standard) and POST /chat/stream (SSE).
No AI logic here — only HTTP concerns: parsing, response formatting, error handling.
"""

import structlog
from pydantic import ValidationError

# Imports APIRouter to structure routes modularly and
# HTTPException to return error responses to the client.
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse

from src.models.state_models import OrchestrationState
from src.models.chat_models import ChatRequest, ChatResponse
from src.agent.runner import run_agent, stream_agent
from src.api.dependencies import get_agent

# Imports the business logic function that handles the AI processing (RAG/SQL)
from src.tools.search.vector_search import perform_vector_search

# Initializes a logger for this module, using the module name to trace where logs originate.
# logger = logging.getLogger(__name__)

# Creates a router instance. All routes defined here will start with /api/chat and
# be grouped under "ai-chat" in the auto-generated documentation.
router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = structlog.get_logger(__name__)

# TODO: validate inputs on .net backend
# Defines the function that handles the request.
# FastAPI automatically validates the incoming request body against the ChatRequest model.


@router.post("", response_model=ChatResponse)
async def chat(chat_request: ChatRequest, agent=Depends(get_agent)) -> ChatResponse:
    """
    Main chat endpoint for the .NET backend.
    """
    log = logger.bind(session_id=chat_request.session_id)
    log.info("Chat request", message_preview=chat_request.message[:80])

    try:
        state = OrchestrationState(
            session_id=chat_request.session_id,
            user_id=chat_request.user_id,
            user_message=chat_request.message
        )

        final_state = await run_agent(agent, state)

        result = ChatResponse(
            answer=final_state.answer or "sorry, I could not generate a response.",
            session_id=final_state.session_id,
            success=len(final_state.errors) == 0,
            route=final_state.routing_decision.route.value if final_state.routing_decision else None,
            metadata=final_state.metadata
        )

        return result

    except Exception as ex:
        logger.exception("Chat failed", error=str(ex))
        raise HTTPException(
            status_code=500,
            detail="AI service error. Please try again."
        ) from ex


@router.post("/stream")
async def chat_stream(request: ChatRequest, agent=Depends(get_agent)):
    """
    Streaming chat endpoint — returns tokens via Server-Sent Events (SSE).
    Connect from React using EventSource or the fetch API with streaming.
    """
    log = logger.bind(session_id=request.session_id)
    log.info("Stream request", message_preview=request.message[:80])

    # Instead of waiting for the full response, this sends tokens back word by word as the LLM generates them — giving the typing effect in the UI.
    # `async for token in stream_agent(...)` — iterates over tokens as they arrive from the LLM, one by one.
    # `yield f"data: {token}\n\n"` — the `\n\n` is required by the SSE protocol. React reads these chunks and appends them to the message in real time.
    # `yield "data: [DONE]\n\n"` — signals to the React frontend that streaming is finished.
    async def event_generator():
        async for token in stream_agent(agent, request.message, request.session_id):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    result = StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )

    return result


@router.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest):
    # Docstring explaining what this endpoint does (receives a question, processes it, returns an answer).
    """
    Receives a question from .NET, processes it via RAG/SQL fallback,
    and returns the answer.
    """
    try:
        result = await perform_vector_search(request.question)
        return result

    except ValidationError as e:
        logger.warning("Validation error: %s", e)
        # Returns a 422 Unprocessable Entity error to the client with the validation details.
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as ex:
        logger.exception(
            "Unexpected error processing question: %s", request.question)
        raise HTTPException(
            status_code=500, detail="Internal server error") from ex


# Defines a POST endpoint at POST /api/ai/ask.
# It tells FastAPI to validate the outgoing response against the ChatResponse model.
