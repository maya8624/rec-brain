"""
app/core/error_handlers.py

Centralised exception → HTTP response mapping — mirrors .NET ExceptionHandlingMiddleware.

One handler (exception_handler) catches everything that reaches the HTTP boundary.
to_http_response() maps exception type to (status_code, ErrorResponse) using match/case —
the Python equivalent of a C# switch expression.

Note: exceptions inside the LangGraph graph (nodes, tools, services called via ainvoke)
are caught at the node/tool level for graceful degradation and never reach here.
Streaming generators (SSE) also handle their own exceptions — see chat.py/_event_generator
and enquiry_service/stream_draft_response.
"""
import structlog

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.exceptions import (
    AIServiceError,
    BackendClientError,
    BookingServiceError,
    DepositServiceError,
    ToolValidationError,
)

logger = structlog.get_logger(__name__)


class ErrorResponse(BaseModel):
    name: str
    code: int
    message: str
    thread_id: str | None = None


def to_http_response(exc: Exception, thread_id: str | None = None) -> tuple[int, ErrorResponse]:
    """
    Maps exception type to (status_code, ErrorResponse).
    Most specific subclass first — same rule as C# switch expression ordering.
    """
    match exc:
        case ToolValidationError():
            return 422, ErrorResponse(
                name="VALIDATION_ERROR",
                code=422,
                message=str(exc),
                thread_id=thread_id,
            )
        case BookingServiceError():
            return 503, ErrorResponse(
                name="BOOKING_ERROR",
                code=503,
                message=str(exc),
                thread_id=thread_id,
            )
        case DepositServiceError():
            return 503, ErrorResponse(
                name="DEPOSIT_ERROR",
                code=503,
                message=str(exc),
                thread_id=thread_id,
            )
        case BackendClientError():
            return 502, ErrorResponse(
                name="BACKEND_ERROR",
                code=502,
                message="Upstream service error. Please try again.",
                thread_id=thread_id,
            )
        case AIServiceError():
            return 500, ErrorResponse(
                name="AI_SERVICE_ERROR",
                code=500,
                message="AI service error. Please try again.",
                thread_id=thread_id,
            )
        case _:
            return 500, ErrorResponse(
                name="UNEXPECTED_ERROR",
                code=500,
                message="An unexpected error occurred.",
                thread_id=thread_id,
            )


async def exception_handler(request: Request, exc: Exception) -> JSONResponse:
    thread_id = getattr(request.state, "thread_id", None)
    status_code, response = to_http_response(exc, thread_id)
    logger.exception(
        "unhandled_exception",
        path=request.url.path,
        thread_id=thread_id,
        error_name=response.name,
        status_code=status_code,
    )
    return JSONResponse(status_code=status_code, content=response.model_dump())
