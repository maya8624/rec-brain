import logging
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)


def get_llm() -> BaseChatModel:
    """Returns the configured LLM instance based on LLM_PROVIDER setting."""

    # if settings.LLM_PROVIDER == "openai":
    logger.info(
        "Initializing ChatOpenAI LLM: %s",
        settings.OPENAI_MODEL_NAME
    )

    return ChatOpenAI(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        max_retries=2,
    )

    # from langchain_groq import ChatGroq
    # logger.info("Initializing ChatGroq LLM: %s", settings.MODEL_NAME)
    # return ChatGroq(
    #     model=settings.MODEL_NAME,
    #     api_key=settings.GROQ_API_KEY,
    #     temperature=settings.LLM_TEMPERATURE,
    #     max_tokens=settings.LLM_MAX_TOKENS,
    #     max_retries=2,
    # )
