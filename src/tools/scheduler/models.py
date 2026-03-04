"""
Pydantic v2 input schemas for scheduler tools.
LangChain uses these to generate JSON schema for the LLM.
"""

import re
from datetime import date, time
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class AppointmentStatus(str, Enum):
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    RESCHEDULED = "rescheduled"
    PENDING = "pending"


class CheckAvailabilityInput(BaseModel):
    property_id: str = Field(..., description="Unique property identifier")
    preferred_date: date = Field(...,
                                 description="Desired viewing date (YYYY-MM-DD)")
    preferred_time: time = Field(...,
                                 description="Desired viewing time (HH:MM, 24hr)")

    @field_validator("preferred_date")
    @classmethod
    # This prevents booking appointments in the past.
    def date_must_be_future(cls, value: date) -> date:
        from datetime import date as date_type
        if value <= date_type.today():
            raise ValueError("Appointment date must be in the future")
        return value

    @field_validator("preferred_time")
    @classmethod
    def time_must_be_business_hours(cls, value: time) -> time:
        if not (time(9, 0) <= value <= time(18, 0)):
            raise ValueError("Viewing time must be between 09:00 and 18:00")
        return value


class ScheduleViewingInput(BaseModel):
    property_id: str = Field(..., description="Unique property identifier")
    date: date = Field(..., description="Confirmed viewing date (YYYY-MM-DD)")
    time: time = Field(..., description="Confirmed viewing time (HH:MM, 24hr)")
    customer_name: str = Field(..., min_length=2,
                               description="Full name of the customer")
    customer_email: Optional[str] = Field(
        None, description="Customer email address")
    customer_phone: Optional[str] = Field(
        None, description="Customer phone number")
    notes: Optional[str] = Field(
        None, max_length=500, description="Additional notes")

    @field_validator("customer_email")
    @classmethod
    def validate_email(cls, value: Optional[str]) -> Optional[str]:
        if value and not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", value):
            raise ValueError("Invalid email format")
        return value

    @model_validator(mode="after")
    def must_have_contact(self) -> "ScheduleViewingInput":
        if not self.customer_email and not self.customer_phone:
            raise ValueError(
                "At least one of customer_email or customer_phone is required")
        return self


class CancelViewingInput(BaseModel):
    appointment_id: str = Field(...,
                                description="Unique appointment reference ID")
    reason: Optional[str] = Field(
        None, max_length=300, description="Cancellation reason")
    reschedule: bool = Field(
        False, description="Whether this is a reschedule request")
