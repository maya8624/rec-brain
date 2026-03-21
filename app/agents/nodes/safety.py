"""
safety_node — guards against infinite tool-error loops.

Called when a tool invocation fails.  Increments error_count and,
once the threshold is reached, sets requires_human=True so the router
can exit gracefully instead of retrying forever.

The node is idempotent once requires_human is already True: it will
still increment the counter (useful for debugging) but will not
re-trigger any side-effects.
"""

import logging

from app.agents.state import RealEstateAgentState

logger = logging.getLogger(__name__)

# Consecutive tool failures before escalating to a human agent
MAX_ERRORS_BEFORE_ESCALATION = 3


def safety_node(state: RealEstateAgentState) -> dict:
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
        MAX_ERRORS_BEFORE_ESCALATION,
    )

    if new_error_count >= MAX_ERRORS_BEFORE_ESCALATION:
        logger.error(
            "safety_node | escalating to human after %d consecutive errors",
            new_error_count,
        )
        return {
            "error_count": new_error_count,
            "requires_human": True,
        }

    return {"error_count": new_error_count}
