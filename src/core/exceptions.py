class AIServiceError(Exception):
    """Base exception for all AI service errors."""
    pass


class BackendClientError(AIServiceError):
    """Raised when the .NET backend returns an error or is unreachable."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class ToolExecutionError(AIServiceError):
    """Raised when a LangChain tool fails to execute."""

    def __init__(self, tool_name: str, reason: str):
        super().__init__(f"Tool '{tool_name}' failed: {reason}")
        self.tool_name = tool_name
        self.reason = reason


class AgentError(AIServiceError):
    """Raised when the LangGraph agent fails to produce a response."""
    pass


class ValidationError(AIServiceError):
    """Raised when tool input validation fails."""
    pass
