# Imports the ChatOllama class from the langchain_ollama package.
# This class allows us to communicate with a locally running Ollama instance.
# from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from src.config import settings

# Defines a function to get a configured instance of the ChatOllama LLM.
# It type-hints that it will return a 'ChatOllama' object.


def get_llm() -> ChatGroq:
    """Returns the configured ChatOllama LLM instance."""

    # Initializes and returns the ChatOllama object with specific configurations.
    # return ChatOllama(
    #     # The name of the model to use (e.g., "llama3", "mistral") from settings.
    #     model=settings.MODEL_NAME,

    #     # The URL where your local Ollama server is running (e.g., "http://localhost:11434").
    #     base_url=settings.OLLAMA_BASE_URL,
    #     num_thread=8,        # set to your CPU core count
    #     num_predict=500,     # limit max output tokens
    #     # Sets temperature to 0 for deterministic output.
    #     # This is a best practice for generating structured data like SQL,
    #     # as it makes the model less "creative" and more reliable.
    #     temperature=0,
    # )

    llm = ChatGroq(
        model=settings.MODEL_NAME,
        api_key=settings.GROQ_API_KEY,
        temperature=0,
    )

    return llm
