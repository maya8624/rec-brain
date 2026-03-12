from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class UserIntent(str, Enum):
    """
    High-level user intent detected by the hybrid router
    """

    PROPERTY_SEARCH = "property_search"
    PROPERTY_DETAILS = "property_details"
    MARKET_INFO = "market_info"
    CHECK_AVAILABILITY = "check_availability"
    SCHEDULE_VIEWING = "schedule_viewing"
    CANCEL_VIEWING = "cancel_viewing"
    GENERAL_CHAT = "general_chat"
    UNKNOWN = "unknown"


class RouteType(str, Enum):
    """
    Execution path selected by the hybrid router.
    """

    SQL_ONLY = "sql_only"
    VECTOR_ONLY = "vector_only"
    TOOL_ONLY = "tool_only"
    SQL_TOOL = "sql_tool"
    VECTOR_SQL = "vector_sql"
    VECTOR_TOOL = "vector_tool"
    VECTOR_SQL_TOOL = "vector_sql_tool"
    GENERAL = "general"
    UNKNOWN = "unknown"


class RoutingDecision(BaseModel):
    """
    Stuctured output of the hybrid router.
    This model tells downstream components what kind of request
    was detected and which execution paths should be used.
    """
    intent: UserIntent = Field(
        ...,
        description="Detected high-level user intent."
    )

    route: RouteType = Field(
        ...,
        description="Selected execution route for this request."
    )

    use_sql: bool = Field(
        default=False,
        description="Whether structured SQL retrieval should be used."
    )

    use_vector: bool = Field(
        default=False,
        description="Whether semantic/vector retrieval should be used."
    )

    use_tools: bool = Field(
        default=False,
        description="Whether tool calling should be used."
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Router confidence score betwen 0.0 and 1.0."
    )

    candidate_tools: list[str] = Field(
        default_factory=list,
        description="List of possible tool names relevant to the request."
    )

    entities: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured entities extracted from the user message."
    )

    resoning: str | None = Field(
        default=None,
        description="Optional short explanation of why this route was selected."
    )
