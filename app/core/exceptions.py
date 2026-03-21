"""
Shared exceptions for the AI service.
"""


class AIServiceError(Exception):
    """Base exception for all AI service errors."""


class BackendClientError(AIServiceError):
    """Low-level HTTP failure from BackendClient — carries status_code."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class BookingServiceError(AIServiceError):
    """Backend or connectivity failure — show user-friendly message."""


class BookingValidationError(BookingServiceError):
    """Invalid input data — message is safe to surface to the user."""


class ToolExecutionError(AIServiceError):
    """Raised when a LangChain tool fails to execute."""

    def __init__(self, tool_name: str, reason: str):
        super().__init__(f"Tool '{tool_name}' failed: {reason}")
        self.tool_name = tool_name
        self.reason = reason


class AgentError(AIServiceError):
    """Raised when the LangGraph agent fails to produce a response."""


class ValidationError(AIServiceError):
    """Raised when tool input validation fails."""


def raise_for_booking_status(e: BackendClientError, confirmation_id: str | None = None) -> None:
    """Map common .NET HTTP error codes to BookingServiceError."""
    status = e.response.status_code

    if status == 404 and confirmation_id:
        raise BookingServiceError(
            f"Booking {confirmation_id} not found. Please check the reference."
        ) from e

    if status == 409 and confirmation_id:
        raise BookingServiceError(
            f"Booking {confirmation_id} cannot be cancelled — already cancelled or completed."
        ) from e

    if status == 409:
        raise BookingServiceError(
            "That time slot is no longer available. Please choose another."
        ) from e

    if status == 422:
        detail = e.response.json().get("detail", "Invalid booking details")
        raise BookingValidationError(detail) from e

    raise BookingServiceError(f"Request failed (HTTP {status}).") from e
