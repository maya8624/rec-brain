"""
app/schemas/property.py

Pydantic models for property data returned by the SQL agent.
Used by the documents router and as structured output hints.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class PropertySummary(BaseModel):
    """Lightweight property summary returned in search results."""
    property_id: str
    address: str
    suburb: str
    price: float = Field(description="AUD — purchase price or weekly rent")
    bedrooms: int = 0
    bathrooms: int = 0
    property_type: str = Field(
        description="house | apartment | townhouse | unit | villa")
    agent_name: str = ""
    agent_phone: str = ""


class PropertySearchResult(BaseModel):
    """Response shape for property search results."""
    success: bool
    properties: list[PropertySummary] = Field(default_factory=list)
    result_count: int = 0
    query_used: str = ""
    error: Optional[str] = None


class SearchResult(BaseModel):
    """Return type for SqlViewService search methods."""
    success: bool
    output: list[dict] | None = None
    result_count: int = 0
    error: str | None = None
