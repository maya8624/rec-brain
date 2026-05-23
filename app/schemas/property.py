"""
app/schemas/property.py

Pydantic models for property data returned by the SQL agent.
Used by the documents router and as structured output hints.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field


class Listing(BaseModel):
    """A row from v_listings — full column set returned by SqlViewService."""
    listing_id: UUID
    property_id: UUID
    listing_type: str
    listing_status: str
    price: Decimal
    bedrooms: int = 0
    bathrooms: int = 0
    car_spaces: int = 0
    pet_friendly: bool = False
    property_type: str
    title: str = ""
    description: str | None = None
    address_line1: str = ""
    address_line2: str | None = None
    suburb: str = ""
    state: str = ""
    postcode: str = ""
    available_from_utc: datetime | None = None
    land_size_sqm: Decimal | None = None
    building_size_sqm: Decimal | None = None
    year_built: int | None = None
    image_url: str | None = None
    agent_first_name: str = ""
    agent_last_name: str = ""
    agent_email: str = ""
    agent_phone: str = ""
    agency_name: str = ""
    agency_phone: str = ""


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
