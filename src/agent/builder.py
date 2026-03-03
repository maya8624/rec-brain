"""
Agent builder — constructs the LangGraph ReAct agent.
Separated from runner.py so the agent can be built once and reused.
"""

import structlog
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from src.core.config import settings
from agent.prompt import MASTER_SYSTEM_PROMPT
from tools import all_tools

logger = structlog.get_logger(__name__)


def build_llm() -> ChatGroq:
    return ChatGroq(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        api_key=settings.GROQ_API_KEY,
        streaming=True,
        max_tokens=settings.LLM_MAX_TOKENS
    )


def build_agent():
    """
    Build and return the LangGraph ReAct agent.
    Call once at app startup and store on app.state.

    For production: swap MemorySaver for PostgresSaver to persist
    conversation history across service restarts.
    """
    llm = build_llm()  # Initialize the configured LLM client used by the agent.
    # Keep conversation/checkpoint state in process memory.
    memory = MemorySaver()

    agent = create_react_agent(  # Build a ReAct agent that can reason and call tools.
        model=llm,  # Use the LLM instance created above.
        tools=all_tools,  # Register every tool the agent is allowed to call.
        # Persist graph state between steps/turns in memory.
        checkpointer=memory,
        # Apply global system instructions to state.
        state_modifier=MASTER_SYSTEM_PROMPT,
    )  # Final configured agent instance.

    logger.info(
        "Agent built",
        model=settings.LLM_MODEL,
        tools=[t.name for t in all_tools],
    )
    return agent
