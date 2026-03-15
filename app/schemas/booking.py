"""
app/schemas/booking.py

Pydantic models for inspection booking data.
These mirror the .NET backend contract for availability,
booking, and cancellation endpoints.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class AvailableSlot(BaseModel):
    """A single available inspection time slot from .NET."""
    datetime: str = Field(description="YYYY-MM-DD HH:MM format")
    agent_name: str = ""
    available: bool = True


class BookingConfirmation(BaseModel):
    """Confirmation details returned after a successful booking."""
    confirmation_id: str
    property_address: str
    confirmed_datetime: str
    agent_name: str = ""
    agent_phone: str = ""


class CancellationConfirmation(BaseModel):
    """Confirmation returned after a successful cancellation."""
    confirmation_id: str
    message: str
