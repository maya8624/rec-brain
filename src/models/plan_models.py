from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StepType(str, Enum):
    """
    Types of execution step produced by the planner. What kind of work each step performs.
    """

    SQL_SEARCH = "sql_search"
    VECTOR_SEARCH = "vector_search"
    TOOL_CALL = "tool_call"
    RESPONSE_ONLY = "response_only"


class PlanStep(BaseModel):
    """
    A single step in the  execution plan
    """
    step_number: int = Field(
        ...,
        ge=1
    )

    step_type: StepType = Field(
        ...,
        description="Type of execution this step represents."
    )

    description: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description="Human-readable description of what this step does."
    )

    query: str | None = Field(
        default=None,
        description="Query to use for SQL or vector retrieval steps."
    )

    tool_name: str | None = Field(
        default=None,
        description="Tool name for tool execution steps."
    )

    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured inputs required for this step."
    )

    depends_on: list[int] = Field(
        default_factory=list,
        description="List of prior step numbers that must complete before this step."
    )


class ExecutionPlan(BaseModel):
    """
    Full plan created by the planner for the current user request.
    """

    goal: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description="Planner's concise summary of the user's goal"
    )

    steps: list[PlanStep] = Field(
        default_factory=list,
        description="Ordered execution steps to perform."
    )

    requires_clarification: bool = Field(
        default=False,
        description="Whether the system needs more information before execution."
    )

    clarification_question: str | None = Field(
        default=None,
        description="Question to ask the user when clarification is needed."
    )
