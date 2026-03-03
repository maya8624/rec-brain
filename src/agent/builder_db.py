"""Agent builder with Postgres checkpointing for multi-server deployments."""

import structlog  # Structured logging used across the service.
from langchain_groq import ChatGroq  # Groq chat model client.
from langgraph.checkpoint.memory import MemorySaver  # In-memory fallback for local/dev runs.
from langgraph.checkpoint.postgres import PostgresSaver  # Durable checkpointer backed by Postgres.
from langgraph.prebuilt import create_react_agent  # Helper to build a ReAct-style LangGraph agent.

from src.core.config import settings  # Application configuration from environment variables.
from agent.prompt import MASTER_SYSTEM_PROMPT  # Global system prompt injected into agent state.
from tools import all_tools  # Tool registry exposed to the agent.

logger = structlog.get_logger(__name__)  # Module-scoped logger instance.


def build_llm() -> ChatGroq:
    """Create and return the configured Groq chat model."""
    return ChatGroq(
        model=settings.LLM_MODEL,  # Model name to use (for example: llama-3.3-70b-versatile).
        temperature=settings.LLM_TEMPERATURE,  # Sampling temperature for response creativity.
        api_key=settings.GROQ_API_KEY,  # API key used to authenticate with Groq.
        streaming=True,  # Stream tokens as they are generated.
        max_tokens=settings.LLM_MAX_TOKENS,  # Upper bound for generated tokens per response.
    )


def build_checkpointer():
    """Return (checkpointer, checkpointer_cm) based on environment mode."""
    if settings.PRODUCTION:  # Use durable storage in production environments.
        checkpointer_cm = PostgresSaver.from_conn_string(
            settings.DATABASE_URL
        )  # Create a context manager from the Postgres connection string.
        checkpointer = checkpointer_cm.__enter__()  # Open the checkpointer resources/connection.
        checkpointer.setup()  # Ensure required checkpoint tables/schema exist.
        return checkpointer, checkpointer_cm  # Return both active saver and its context manager.
    return MemorySaver(), None  # Use process memory in non-production and no context manager.


def build_agent():
    """
    Build and return the LangGraph ReAct agent with its checkpointer context manager.

    Returns:
        tuple: (agent, checkpointer_cm)
            agent: Configured ReAct agent instance.
            checkpointer_cm: Context manager to close on shutdown (None for MemorySaver).
    """
    llm = build_llm()  # Initialize the LLM used by the agent.
    checkpointer, checkpointer_cm = build_checkpointer()  # Select Postgres or in-memory state store.

    agent = create_react_agent(
        model=llm,  # Attach LLM for reasoning/response generation.
        tools=all_tools,  # Register all callable tools for tool-use steps.
        checkpointer=checkpointer,  # Persist and restore graph state between turns.
        state_modifier=MASTER_SYSTEM_PROMPT,  # Apply global instruction context to state.
    )

    logger.info(
        "Agent built",  # Log successful agent construction.
        model=settings.LLM_MODEL,  # Log model identifier for observability.
        tools=[t.name for t in all_tools],  # Log registered tool names.
        checkpointer=type(checkpointer).__name__,  # Log storage backend type in use.
    )
    return agent, checkpointer_cm  # Return agent and lifecycle handle for cleanup.
