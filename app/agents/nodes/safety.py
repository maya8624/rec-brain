"""
safety_node — guards against infinite tool-error loops.

Called when a tool invocation fails. Increments error_count and,
once the threshold is reached, sets requires_human=True so the router
can exit gracefully instead of retrying forever.
"""

import logging
from typing import Any

from app.agents.state import RealEstateAgentState
from app.core.config import settings

logger = logging.getLogger(__name__)


def safety_node(state: RealEstateAgentState) -> dict[str, Any]:
    """
    Increment error_count.  If the threshold is reached, set
    requires_human=True so the router ends the turn gracefully.

    Returns partial state: { error_count } or { error_count, requires_human }.
    """
    current_errors = state.get("error_count", 0)
    new_error_count = current_errors + 1

    logger.warning(
        "safety_node | error_count %d -> %d (threshold=%d)",
        current_errors,
        new_error_count,
        settings.MAX_ERRORS_BEFORE_ESCALATION,
    )

    if new_error_count >= settings.MAX_ERRORS_BEFORE_ESCALATION:
        logger.error(
            "safety_node | escalating to human after %d consecutive errors",
            new_error_count,
        )

        return {
            "error_count": min(new_error_count, settings.MAX_ERRORS_BEFORE_ESCALATION),
            "requires_human": True,
        }

    return {"error_count": new_error_count}
