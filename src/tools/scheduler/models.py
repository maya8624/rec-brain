"""
Pydantic v2 input schemas for scheduler tools.
LangChain uses these to generate JSON schema for the LLM.

IMPORTANT: Field descriptions are read by Groq to know what format to send.
Be explicit — vague descriptions cause format errors.
"""

from datetime import date, time
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator
import re


class AppointmentStatus(str, Enum):
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    RESCHEDULED = "rescheduled"
    PENDING = "pending"


class CheckAvailabilityInput(BaseModel):
    property_id: str = Field(
        ...,
        description="Unique property identifier e.g. 'PROP-001'"
    )
    preferred_date: str = Field(
        ...,
        description=(
            "Viewing date in YYYY-MM-DD format only. "
            "You MUST convert relative dates like 'this Saturday' or 'tomorrow' "
            "to an actual date before calling this tool. "
            "Example: '2025-03-08'"
        )
    )
    preferred_time: str = Field(
        ...,
        description=(
            "Viewing time in HH:MM 24-hour format only. "
            "Example: '10:00', '14:30'. "
            "Must be between 09:00 and 18:00."
        )
    )

    @field_validator("preferred_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        from datetime import datetime, date as date_type
        try:
            parsed = datetime.strptime(v, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError(
                f"Date must be in YYYY-MM-DD format, got '{v}'. "
                "Convert relative dates like 'this Saturday' to a real date first."
            )
        if parsed <= date_type.today():
            raise ValueError(f"Date must be in the future, got '{v}'")
        return v

    @field_validator("preferred_time")
    @classmethod
    def validate_time_format(cls, v: str) -> str:
        from datetime import datetime, time as time_type
        try:
            parsed = datetime.strptime(v, "%H:%M").time()
        except ValueError:
            raise ValueError(
                f"Time must be in HH:MM 24-hour format, got '{v}'. "
                "Example: '10:00', '14:30'."
            )
        if not (time_type(9, 0) <= parsed <= time_type(18, 0)):
            raise ValueError(
                f"Time must be between 09:00 and 18:00, got '{v}'"
            )
        return v


class ScheduleViewingInput(BaseModel):
    property_id: str = Field(
        ...,
        description="Unique property identifier e.g. 'PROP-001'"
    )
    date: str = Field(
        ...,
        description=(
            "Confirmed viewing date in YYYY-MM-DD format only. "
            "Example: '2025-03-08'"
        )
    )
    time: str = Field(
        ...,
        description=(
            "Confirmed viewing time in HH:MM 24-hour format only. "
            "Example: '10:00', '14:30'."
        )
    )
    customer_name: str = Field(
        ...,
        min_length=2,
        description="Full name of the customer e.g. 'John Smith'"
    )
    customer_email: Optional[str] = Field(
        None,
        description="Customer email address e.g. 'john@example.com'"
    )
    customer_phone: Optional[str] = Field(
        None,
        description="Customer phone number e.g. '0412 345 678'"
    )
    notes: Optional[str] = Field(
        None,
        max_length=500,
        description="Any additional notes for the agent"
    )

    @field_validator("date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        from datetime import datetime
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(
                f"Date must be in YYYY-MM-DD format, got '{v}'"
            )
        return v

    @field_validator("time")
    @classmethod
    def validate_time_format(cls, v: str) -> str:
        from datetime import datetime
        try:
            datetime.strptime(v, "%H:%M")
        except ValueError:
            raise ValueError(
                f"Time must be in HH:MM 24-hour format, got '{v}'"
            )
        return v

    @field_validator("customer_email")
    @classmethod
    def validate_email(cls, v: Optional[str]) -> Optional[str]:
        if v and not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", v):
            raise ValueError("Invalid email format")
        return v

    @model_validator(mode="after")
    def must_have_contact(self) -> "ScheduleViewingInput":
        if not self.customer_email and not self.customer_phone:
            raise ValueError(
                "At least one of customer_email or customer_phone is required"
            )
        return self


class CancelViewingInput(BaseModel):
    appointment_id: str = Field(
        ...,
        description="Appointment reference ID from the confirmation e.g. 'APT-1001'"
    )
    reason: Optional[str] = Field(
        None,
        max_length=300,
        description="Reason for cancellation"
    )
    reschedule: bool = Field(
        False,
        description="Set to true if customer wants to reschedule instead of cancel"
    )
