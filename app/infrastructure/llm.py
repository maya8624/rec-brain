import logging
from langchain_groq import ChatGroq
from app.core.config import settings

logger = logging.getLogger(__name__)


def get_llm() -> ChatGroq:
    """Returns the configured ChatGroq LLM instance."""

    logger.info("Initializing ChatGroq LLM: %s", settings.MODEL_NAME)

    return ChatGroq(
        model=settings.MODEL_NAME,
        api_key=settings.GROQ_API_KEY,
        temperature=settings.LLM_TEMPERATURE,
        max_retries=3,
    )
