"""
app/schemas/booking.py

Pydantic models for inspection booking data.
These mirror the .NET backend contract for availability,
booking, and cancellation endpoints.
"""
from datetime import datetime
from pydantic import BaseModel, ConfigDict, EmailStr, Field, TypeAdapter, computed_field, field_validator
from app.core.config import DATETIME_FORMAT
from app.core.utils import fmt_dt_sydney


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


class BookingResult(BaseModel):
    """Tool return for book_inspection."""
    success: bool
    confirmation_id: str = ""
    property_id: str = ""
    property_address: str = ""
    start_at_utc: datetime | None = None
    end_at_utc: datetime | None = None
    agent_name: str = ""
    agent_phone: str = ""
    message: str = ""
    error: str | None = None


class BookingLookupResult(BaseModel):
    """Booking confirmation from .NET; also the get_booking tool return."""
    success: bool = True
    confirmation_id: str = ""
    property_id: str = ""
    property_address: str = ""
    status: str = ""
    agent_first_name: str = Field("", exclude=True)
    agent_last_name: str = Field("", exclude=True)
    agent_phone: str | None = None
    start_at_utc: datetime | None = Field(None, exclude=True)
    end_at_utc: datetime | None = Field(None, exclude=True)
    bookings: list[dict] = []
    error: str | None = None

    @computed_field
    @property
    def agent_name(self) -> str:
        return f"{self.agent_first_name} {self.agent_last_name}".strip()

    @computed_field
    @property
    def start_at(self) -> str:
        return fmt_dt_sydney(self.start_at_utc)

    @computed_field
    @property
    def end_at(self) -> str:
        return fmt_dt_sydney(self.end_at_utc)
