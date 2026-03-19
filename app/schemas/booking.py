"""
app/schemas/booking.py

Pydantic models for inspection booking data.
These mirror the .NET backend contract for availability,
booking, and cancellation endpoints.
"""
from pydantic import BaseModel, EmailStr, Field, field_validator
from app.core.config import DATETIME_FORMAT
from datetime import datetime


class ContactInfo(BaseModel):
    name: str = Field(min_length=2)
    email: EmailStr
    phone: str

    @field_validator("phone")
    @classmethod
    def phone_has_enough_digits(cls, v: str) -> str:
        if len([c for c in v if c.isdigit()]) < 8:
            raise ValueError("contact_phone must be a valid phone number")
        return v


class BookingRequest(BaseModel):
    property_id: str = Field(min_length=1)
    datetime_slot: str  # validated below
    contact: ContactInfo

    @field_validator("datetime_slot")
    @classmethod
    def slot_must_be_future(cls, v: str) -> str:
        try:
            dt = datetime.strptime(v, DATETIME_FORMAT)
        except ValueError:
            raise ValueError(f"Expected YYYY-MM-DD HH:MM, got '{v}'")
        if dt <= datetime.now():
            raise ValueError("Inspection datetime must be in the future")
        return v


class AvailableSlot(BaseModel):
    """A single available inspection time slot from .NET."""
    datetime: str = Field(description="YYYY-MM-DD HH:MM format")
    agent_name: str = ""
    available: bool = True


class AvailabilityResult(BaseModel):
    """Returned availability result or error"""
    success: bool
    property_id: str
    available_slots: list[AvailableSlot] = []
    slot_count: int = 0
    error: str | None = None


class BookingConfirmation(BaseModel):
    """Confirmation details returned after a successful booking."""
    confirmation_id: str
    property_address: str
    confirmed_datetime: str
    agent_name: str = ""
    agent_phone: str = ""


class CancellationRequest(BaseModel):
    confirmation_id: str = Field(min_length=1)
    reason: str | None = None


class CancellationConfirmation(BaseModel):
    """Confirmation returned after a successful cancellation."""
    confirmation_id: str
    message: str = "Booking successfully cancelled"
