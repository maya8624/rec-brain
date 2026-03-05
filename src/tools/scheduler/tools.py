"""
Scheduler LangChain tools.
Each tool is a single responsibility class — no business logic, only:
  1. Call the backend service
  2. Format the result as an LLM-readable string
  3. Handle errors gracefully (never expose raw exceptions to the LLM)
"""

from datetime import date, time
from typing import Optional, Type

import structlog
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from services.backend_client import backend_client
from tools.scheduler.models import (
    CheckAvailabilityInput,
    ScheduleViewingInput,
    CancelViewingInput,
    AppointmentStatus,
)

logger = structlog.get_logger(__name__)


class CheckAvailabilityTool(BaseTool):
    """
    Tool: check_availability
    ------------------------
    Checks whether a specific property is available for a viewing
    on a given date and time by querying the .NET backend API.

    When to use:
        ALWAYS call this before schedule_viewing. Never skip this step.
        If the slot is taken, returns up to 3 alternative slots.

    Returns:
        A plain string the LLM reads to decide the next action.
        Either a confirmation with the assigned agent's name,
        or a list of alternative available slots.

    Raises:
        Nothing — all exceptions are caught and returned as
        a friendly error string so the agent loop never breaks.
    """

    name: str = "check_availability"
    description: str = (
        "Check if a property is available for viewing on a specific date and time. "
        "ALWAYS call this before schedule_viewing. "
        "Returns available slots if the requested time is taken."
    )
    args_schema: Type[BaseModel] = CheckAvailabilityInput

    async def _arun(
        self,
        property_id: str,
        preferred_date: date,
        preferred_time: time,
    ) -> str:
        """
        Execute the availability check.

        Args:
            property_id:    Unique identifier of the property (e.g. "PROP-001")
            preferred_date: The date the customer wants to view (must be future)
            preferred_time: The time the customer wants to view (9AM–6PM only)

        Returns:
            Human-readable string for the LLM describing availability
            and next steps, or an error message if the call fails.
        """
        log = logger.bind(
            tool=self.name,
            property_id=property_id,
            date=str(preferred_date)
        )

        try:
            data = await backend_client.get(
                path="/api/appointments/availability",
                params={
                    "propertyId": property_id,
                    "date": preferred_date.isoformat(),
                    "time": preferred_time.strftime("%H:%M"),
                },
            )

            if data.get("available"):
                agent = data.get("agent", {})
                log.info("Slot available", agent=agent.get("name"))

                return (
                    f"✅ Available! {preferred_date.strftime('%A, %d %B %Y')} at "
                    f"{preferred_time.strftime('%I:%M %p')} is open. "
                    f"Assigned agent: {agent.get('name', 'TBD')}. "
                    f"Proceed with schedule_viewing."
                )
            else:
                alternatives = data.get("alternativeSlots", [])
                alt_text = ""

                if alternatives:
                    slots = [
                        f"  • {s['date']} at {s['time']} (Agent: {s.get('agent', 'TBD')})"
                        for s in alternatives[:3]
                    ]
                    alt_text = "\n\nAlternative slots:\n" + "\n".join(slots)

                log.info("Slot unavailable", alternatives=len(alternatives))
                return f"❌ That slot is not available.{alt_text}"

        except Exception as e:
            log.error("Tool failed", error=str(e))
            return (
                "⚠️ Could not check availability right now. "
                "Please try a different date or contact the agency directly."
            )

    def _run(self, *args, **kwargs):
        """Sync version is intentionally disabled — this service is fully async."""
        raise NotImplementedError("Use async _arun")


class ScheduleViewingTool(BaseTool):
    """
    Tool: schedule_viewing
    ----------------------
    Books a confirmed property viewing appointment by posting
    the customer's details to the .NET backend API.

    When to use:
        Only call this AFTER check_availability confirms the slot is open.
        Never call without a prior successful availability check.

    Requires:
        property_id, date, time, customer_name, and at least one of
        customer_email or customer_phone (enforced by ScheduleViewingInput).

    Returns:
        A formatted confirmation string with appointment details and
        reference ID, or a friendly error message if booking fails.

    Raises:
        Nothing — all exceptions are caught and returned as
        a friendly error string so the agent loop never breaks.
    """

    name: str = "schedule_viewing"
    description: str = (
        "Book a confirmed property viewing appointment. "
        "Only call AFTER check_availability confirms the slot is open. "
        "Requires: property_id, date, time, customer_name, and customer_email OR customer_phone."
    )
    args_schema: Type[BaseModel] = ScheduleViewingInput

    async def _arun(
        self,
        property_id: str,
        date: date,
        time: time,
        customer_name: str,
        customer_email: Optional[str] = None,
        customer_phone: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> str:
        """
        Execute the appointment booking.

        Args:
            property_id:    Unique identifier of the property
            date:           Confirmed viewing date
            time:           Confirmed viewing time
            customer_name:  Full name of the customer
            customer_email: Customer email address (optional if phone provided)
            customer_phone: Customer phone number (optional if email provided)
            notes:          Any additional notes for the agent (optional, max 500 chars)

        Returns:
            Formatted confirmation string with property address, date, time,
            agent name, and reference ID. Or a friendly error message.
        """
        log = logger.bind(
            tool=self.name,
            property_id=property_id,
            customer_name=customer_name
        )

        try:
            data = await backend_client.post(
                path="/api/appointments",
                body={
                    "propertyId": property_id,
                    "date": date.isoformat(),
                    "time": time.strftime("%H:%M"),
                    "customerName": customer_name,
                    "customerEmail": customer_email,
                    "customerPhone": customer_phone,
                    "notes": notes,
                },
            )

            appt = data.get("appointment", {})
            appt_id = appt.get("id", "N/A")
            contact = customer_email or customer_phone

            log.info("Appointment booked", appointment_id=appt_id)

            return (
                f"✅ Appointment confirmed!\n"
                f"- 📍 Property: {appt.get('propertyAddress', property_id)}\n"
                f"- 📅 Date: {date.strftime('%A, %d %B %Y')}\n"
                f"- 🕐 Time: {time.strftime('%I:%M %p')}\n"
                f"- 👤 Agent: {appt.get('agentName', 'TBD')}\n"
                f"- 🔖 Reference: {appt_id}\n\n"
                f"Confirmation sent to {contact}."
            )

        except Exception as e:
            log.error("Tool failed", error=str(e))
            return (
                "⚠️ Failed to book the appointment. "
                "Please try again or contact the agency directly."
            )

    def _run(self, *args, **kwargs):
        """Sync version is intentionally disabled — this service is fully async."""
        raise NotImplementedError("Use async _arun")


class CancelViewingTool(BaseTool):
    """
    Tool: cancel_viewing
    --------------------
    Cancels or flags an existing appointment for rescheduling
    by patching the appointment status in the .NET backend API.

    When to use:
        When the customer wants to cancel or reschedule a booking.
        Ask for the appointment reference ID (e.g. "APT-9982") first.
        If they don't have it, ask for their name + property address.

    Rescheduling flow:
        Set reschedule=True to flag for rescheduling instead of cancelling.
        The tool will then guide the LLM to start a fresh scheduling flow
        using check_availability followed by schedule_viewing.

    Returns:
        A confirmation string for cancellation, or rescheduling
        instructions that guide the LLM's next steps.

    Raises:
        Nothing — all exceptions are caught and returned as
        a friendly error string so the agent loop never breaks.
    """

    name: str = "cancel_viewing"
    description: str = (
        "Cancel or reschedule an existing property viewing. "
        "Use when the customer wants to cancel or change their booking. "
        "Requires the appointment_id from their confirmation."
    )
    args_schema: Type[BaseModel] = CancelViewingInput

    async def _arun(
        self,
        appointment_id: str,
        reason: Optional[str] = None,
        reschedule: bool = False,
    ) -> str:
        """
        Execute the cancellation or reschedule flag.

        Args:
            appointment_id: The reference ID from the booking confirmation (e.g. "APT-9982")
            reason:         Optional reason for cancellation (max 300 chars)
            reschedule:     If True, flags for rescheduling instead of cancelling.
                            The LLM will then be guided to start a new booking flow.

        Returns:
            Cancellation confirmation string, or rescheduling instructions
            that tell the LLM to use check_availability + schedule_viewing next.
        """
        log = logger.bind(
            tool=self.name,
            appointment_id=appointment_id,
            reschedule=reschedule
        )

        try:
            await backend_client.patch(
                path=f"/api/appointments/{appointment_id}/status",
                body={
                    "status": AppointmentStatus.RESCHEDULED if reschedule else AppointmentStatus.CANCELLED,
                    "reason": reason,
                },
            )

            log.info("Appointment updated")

            if reschedule:
                return (
                    f"✅ Appointment {appointment_id} flagged for rescheduling. "
                    f"Please ask for a new preferred date and time, "
                    f"then use check_availability followed by schedule_viewing."
                )

            return (
                f"✅ Appointment {appointment_id} has been cancelled. "
                f"A confirmation has been sent to the customer."
            )

        except Exception as e:
            log.error("Tool failed", error=str(e))
            return (
                f"⚠️ Could not cancel appointment {appointment_id}. "
                f"Please ask the customer to call the agency directly."
            )

    def _run(self, *args, **kwargs):
        """Sync version is intentionally disabled — this service is fully async."""
        raise NotImplementedError("Use async _arun")
