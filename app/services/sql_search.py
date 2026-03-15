import asyncio
import logging

from app.core.config import settings
from app.infrastructure.sql_agent import create_real_estate_sql_agent
from app.infrastructure.sql_inspector import track_sql_steps

logger = logging.getLogger(__name__)


async def perform_sql_search(question: str) -> dict[str, str] | None:
    if not question or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    agent_executor = create_real_estate_sql_agent()

    # Schema -> Translation -> Execution -> Formatting
    try:
        response = await asyncio.wait_for(
            agent_executor.ainvoke({"input": question}),
            timeout=settings.AGENT_TIMEOUT_SECONDS
        )
        output = response.get("output", "")
        intermediate_steps = response.get("intermediate_steps", [])

        logger.info("Agent answer: %s", output)
        logger.debug("Intermediate steps: %s", intermediate_steps)

        # Write the background SQL to your log file
        track_sql_steps(response)

        # Log the background SQL for your records
        # log_agent_steps(response)

        return {
            "answer": output,
            "steps": str(intermediate_steps),
        }

    except Exception:
        logger.exception("SQL agent failed for question: %s", question)
        return {"success": False, "error": "SQL agent failed"}
