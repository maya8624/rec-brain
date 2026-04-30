"""
app/schemas/booking.py

Pydantic models for inspection booking data.
These mirror the .NET backend contract for availability,
booking, and cancellation endpoints.
"""
from datetime import datetime
from pydantic import BaseModel, ConfigDict, EmailStr, Field, TypeAdapter, computed_field, field_validator
from app.core.config import DATETIME_FORMAT


class ContactInfo(BaseModel):
    name: str = Field(min_length=2)
    email: EmailStr
    phone: str


class BookingRequest(BaseModel):
    slot_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    notes: str = Field("", max_length=1000)


class AvailableSlot(BaseModel):
    """Maps .NET InspectionSlotDto — field aliases handle camelCase from the API."""
    model_config = ConfigDict(populate_by_name=True)

    slot_id:    str = Field("", alias="id")
    start_at:   str = Field("", alias="startAtUtc")
    end_at:     str = Field("", alias="endAtUtc")
    agent_id:   str = Field("", alias="agentId")
    # agent_name: str = ""
    capacity:   int = 0
    status:     str = ""
    notes:      str = ""

    @field_validator("notes", mode="before")
    @classmethod
    def _notes_none_to_empty(cls, v: object) -> object:
        return v if v is not None else ""

    @computed_field
    @property
    def available(self) -> bool:
        return str(self.status).lower() == "open" and self.capacity > 0


AvailableSlotList = TypeAdapter(list[AvailableSlot])


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
    property_id: str = ""
    property_address: str = ""
    status: str = ""
    agent_first_name: str = ""
    agent_last_name: str = ""
    agent_phone: str | None = None
    start_at_utc: datetime | None = None
    end_at_utc: datetime | None = None


class BookingResult(BaseModel):
    """Tool return for book_inspection."""
    success: bool
    confirmation_id: str = ""
    property_address: str = ""
    start_at_utc: datetime | None = None
    end_at_utc: datetime | None = None
    agent_name: str = ""
    agent_phone: str = ""
    message: str = ""
    error: str | None = None


class CancellationConfirmation(BaseModel):
    """Confirmation returned after a successful cancellation."""
    id: str
    success: bool
    message: str = "Your inspection booking has been successfully cancelled. A confirmation email will be sent to you shortly."


class CancellationResult(BaseModel):
    """Tool return for cancel_inspection."""
    success: bool
    id: str = ""
    message: str = ""
    error: str | None = None


class BookingLookupResult(BaseModel):
    """Tool return for get_booking."""
    success: bool
    confirmation_id: str = ""
    property_id: str = ""
    property_address: str = ""
    status: str = ""
    agent_name: str = ""
    agent_phone: str | None = None
    start_at: str = ""
    end_at: str = ""
    bookings: list[dict] = []   # populated when returning all user bookings
    error: str | None = None
