from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

FingerprintCategory = Literal[
    "DEPENDENCY_FAILURE",
    "NEW_REGRESSION",
    "RECURRING_KNOWN",
    "CONFIG_AUTH",
    "DATA_QUALITY",
    "PERFORMANCE",
]


class ClassifyRequest(BaseModel):
    exception_type: str
    message_template: str
    sample_trace: str
    operation: str


class ClassifyResponse(BaseModel):
    category: FingerprintCategory
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class FingerprintOccurrence(BaseModel):
    occurred_at: datetime
    occurrence_count: int
    rendered_message: str


class FingerprintRow(BaseModel):
    id: str
    level: str
    exception_type: str
    message_template: str
    operation: str
    service_name: str
    category: FingerprintCategory
    first_seen: datetime
    last_seen: datetime
    total_count: int
    sample_trace: str | None = None


class SummarizeRequest(BaseModel):
    fingerprint: FingerprintRow
    occurrences: list[FingerprintOccurrence]


class SummarizeResponse(BaseModel):
    title: str
    body: str
    suggested_fix: str | None
