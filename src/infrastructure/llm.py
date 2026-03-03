# Imports the ChatOllama class from the langchain_ollama package.
# This class allows us to communicate with a locally running Ollama instance.
# from langchain_ollama import ChatOllama
import logging
from functools import lru_cache
from langchain_groq import ChatGroq
from src.api.core.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """Returns the configured ChatOllama LLM instance."""

    logger.info("Initializing ChatGroq LLM: %s", settings.MODEL_NAME)

    return ChatGroq(
        model=settings.MODEL_NAME,
        api_key=settings.GROQ_API_KEY,
        temperature=0,
    )
