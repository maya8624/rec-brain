from typing import Any
from pydantic import BaseModel, Field

from src.models.plan_models import ExecutionPlan
from src.models.routing_models import RoutingDecision


class RetrievalResult(BaseModel):
    """
    Normalize retrieval result for SQL or vector search
    """

    source: str = Field(
        ...,
        description="Source of the retrieval result, for example 'sql' or 'vector'."
    )

    success: bool = Field(
        default=True,
        description="Whether the retrieval step succeeded."
    )

    records: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured records returned by the retrieval step."
    )

    summary: str | None = Field(
        default=None,
        description="Optional short summary of the retrieval result"
    )

    error: str | None = Field(
        default=None,
        description="Error message if the retrieval step failed."
    )


class ToolExecutionResult(BaseModel):
    """
    Normalized result for a tool call.
    """

    tool_name: str = Field(
        ...,
        description="Name of the tool that was executed."
    )

    success: bool = Field(
        default=True,
        description="Whether the tool execution succeeded."
    )

    input: dict[str, Any] = Field(
        default_factory=dict,
        description="Inputs passed to the tool."
    )

    output: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured output returned by the tool."
    )

    error: str | None = Field(
        default=None,
        description="Error message if the tool execution failed."
    )


class OrchestrationState(BaseModel):
    """
   Shared state carried across the orchestration pipeline.
   """
    session_id: str = Field(
        ...,
        description="Conversation session ID provided by the backend."
    )

    user_id: str | None = Field(
        default=None,
        description="Optional user identifier from the backend."
    )

    user_message: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description="Original user message."
    )

    routing_decision: RoutingDecision | None = Field(
        default=None,
        description="Routing decision produced by the hybrid router."
    )

    execution_plan: ExecutionPlan | None = Field(
        default=None,
        description="Execution plan produced by the planner."
    )

    sql_result: RetrievalResult | None = Field(
        default=None,
        description="Normalized result from SQL retrieval."
    )

    vector_result: RetrievalResult | None = Field(
        default=None,
        description="Normalized result from vector retrieval."
    )

    tool_results: list[ToolExecutionResult] = Field(
        default_factory=list,
        description="Results of executed tools."
    )

    final_answer: str | None = Field(
        default=None,
        description="Final composed response returned to the caller."
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional non-sensitive orchestration metadata."
    )

    errors: list[str] = Field(
        default_factory=list,
        description="Collected orchestration errors."
    )

    def add_error(self, error_message: str) -> None:
        """
        Append an error message to the orchestration state.
        """
        if not isinstance(self.errors, list):
            self.errors = []

        self.errors.append(error_message)

    def add_tool_result(self, result: ToolExecutionResult) -> None:
        """
        Append a tool execution result to the orchestration state.
        """
        if not isinstance(self.tool_results, list):
            self.tool_results = []

        self.tool_results.append(result)
