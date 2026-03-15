"""
Mock data for local development.
Simulates responses from the .NET backend API.
Edit this data to test different scenarios.
"""

from datetime import date, timedelta

# ------------------------------------
# Properties
# ------------------------------------

MOCK_PROPERTIES = {
    "PROP-001": {
        "id": "PROP-001",
        "address": "12 Ocean Drive, Bondi Beach NSW 2026",
        "type": "House",
        "bedrooms": 4,
        "bathrooms": 2,
        "price": 2850000,
        "description": "Stunning beachside home with ocean views",
        "agentId": "AGENT-001",
    },
    "PROP-002": {
        "id": "PROP-002",
        "address": "5/88 Crown Street, Surry Hills NSW 2010",
        "type": "Apartment",
        "bedrooms": 2,
        "bathrooms": 1,
        "price": 950000,
        "description": "Modern apartment in the heart of Surry Hills",
        "agentId": "AGENT-002",
    },
    "PROP-003": {
        "id": "PROP-003",
        "address": "34 Maple Avenue, Chatswood NSW 2067",
        "type": "Townhouse",
        "bedrooms": 3,
        "bathrooms": 2,
        "price": 1650000,
        "description": "Spacious townhouse near schools and transport",
        "agentId": "AGENT-001",
    },
}

# ------------------------------------
# Agents
# ------------------------------------

MOCK_AGENTS = {
    "AGENT-001": {"id": "AGENT-001", "name": "Sarah Mitchell", "phone": "0412 345 678"},
    "AGENT-002": {"id": "AGENT-002", "name": "James Park",     "phone": "0423 456 789"},
}

# ------------------------------------
# Availability responses
# ------------------------------------


def get_availability_response(property_id: str, date_str: str, time_str: str) -> dict:
    """
    Simulates .NET availability check response.
    Logic:
      - Weekends at 10:00 → always available
      - Time ending in :30 → unavailable (simulates a taken slot)
      - Everything else → available
    """
    prop = MOCK_PROPERTIES.get(property_id, MOCK_PROPERTIES["PROP-001"])
    agent_id = prop.get("agentId", "AGENT-001")
    agent = MOCK_AGENTS.get(agent_id, MOCK_AGENTS["AGENT-001"])

    # Simulate some slots being taken
    is_taken = time_str.endswith(":30")

    if is_taken:
        future = date.today() + timedelta(days=1)
        return {
            "available": False,
            "alternativeSlots": [
                {
                    "date": future.isoformat(),
                    "time": "10:00",
                    "agent": agent["name"],
                },
                {
                    "date": future.isoformat(),
                    "time": "14:00",
                    "agent": agent["name"],
                },
                {
                    "date": (future + timedelta(days=1)).isoformat(),
                    "time": "11:00",
                    "agent": MOCK_AGENTS["AGENT-002"]["name"],
                },
            ],
        }

    return {
        "available": True,
        "agent": {"id": agent["id"], "name": agent["name"]},
    }


# ------------------------------------
# Appointment responses
# ------------------------------------

_appointment_counter = 1000


def get_create_appointment_response(body: dict) -> dict:
    """Simulates .NET create appointment response."""
    global _appointment_counter
    _appointment_counter += 1

    property_id = body.get("propertyId", "PROP-001")
    prop = MOCK_PROPERTIES.get(property_id, MOCK_PROPERTIES["PROP-001"])
    agent_id = prop.get("agentId", "AGENT-001")
    agent = MOCK_AGENTS.get(agent_id, MOCK_AGENTS["AGENT-001"])

    return {
        "appointment": {
            "id": f"APT-{_appointment_counter}",
            "propertyId": property_id,
            "propertyAddress": prop["address"],
            "date": body.get("date"),
            "time": body.get("time"),
            "customerName": body.get("customerName"),
            "customerEmail": body.get("customerEmail"),
            "customerPhone": body.get("customerPhone"),
            "agentName": agent["name"],
            "status": "confirmed",
        }
    }


def get_cancel_appointment_response(appointment_id: str, body: dict) -> dict:
    """Simulates .NET cancel/reschedule appointment response."""
    return {
        "appointmentId": appointment_id,
        "status": body.get("status", "cancelled"),
        "updatedAt": date.today().isoformat(),
    }
