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
    property_id: str = Field(min_length=1)
    datetime_slot: str  # validated below
    contact: ContactInfo

    @field_validator("datetime_slot")
    @classmethod
    def slot_must_be_future(cls, value: str) -> str:
        try:
            dt = datetime.strptime(value, DATETIME_FORMAT)
        except ValueError:
            raise ValueError(f"Expected YYYY-MM-DD HH:MM, got '{value}'")

        if dt <= datetime.now():
            raise ValueError("Inspection datetime must be in the future")

        return value


class AvailableSlot(BaseModel):
    """Maps .NET InspectionSlotDto — field aliases handle camelCase from the API."""
    model_config = ConfigDict(populate_by_name=True)

    slot_id:    str = Field("", alias="id")
    start_at:   str = Field("", alias="startAtUtc")
    end_at:     str = Field("", alias="endAtUtc")
    agent_id:   str = Field("", alias="agentId")
    agent_name: str = ""
    capacity:   int = Field(0, alias="capacity")
    status:     str = Field("", alias="status")
    notes:      str = Field("", alias="notes")

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
    property_address: str
    confirmed_datetime: str
    agent_name: str = ""
    agent_phone: str = ""


class BookingResult(BaseModel):
    """Tool return for book_inspection."""
    success: bool
    confirmation_id: str = ""
    property_address: str = ""
    confirmed_datetime: str = ""
    agent_name: str = ""
    agent_phone: str = ""
    message: str = ""
    error: str | None = None


class CancellationRequest(BaseModel):
    confirmation_id: str = Field(min_length=1)
    reason: str | None = None


class CancellationConfirmation(BaseModel):
    """Confirmation returned after a successful cancellation."""
    confirmation_id: str
    message: str = "Booking successfully cancelled"


class CancellationResult(BaseModel):
    """Tool return for cancel_inspection."""
    success: bool
    confirmation_id: str = ""
    message: str = ""
    error: str | None = None


class SearchListingResult(BaseModel):
    """Tool return for search_listings."""
    success: bool
    output: str | None = None
    result_count: int = 0
    error: str | None = None
