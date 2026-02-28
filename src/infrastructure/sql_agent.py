import logging
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_core.messages import SystemMessage
from src.infrastructure.database import get_db_wrapper
from src.infrastructure.llm import get_llm

logger = logging.getLogger(__name__)

# 2. Craft a High-Precision System Message
_SYSTEM_MESSAGE = SystemMessage(content=("""
    "You are a Real Estate SQL Expert. Follow these rules strictly:\n"
    "1. ALWAYS use the 'sql_db_list_tables' tool first.\n"
    "2. NEVER use 'SELECT *'. Query specific columns only.\n"
    "3. ALWAYS add 'LIMIT 10'.\n"
    "4. If you cannot find the answer, say 'I couldn't find a direct match in our listings.'\n"
    "5. Do NOT mention table names or SQL technicalities to the customer."
"""))


def create_real_estate_sql_agent():
    logger.info("Creating Real Estate SQL Agent.")

    # Get the base LLM for the Toolkit (solves the 422/Pydantic error)
    raw_llm = get_llm()
    db = get_db_wrapper()

    # Pass raw_llm to toolkit — RunnableRetry is NOT a BaseLanguageModel
    toolkit = SQLDatabaseToolkit(db=db, llm=raw_llm)
    # Get the retry-wrapped LLM for the Agent Executor logic
    agent_llm = raw_llm.with_retry(stop_after_attempt=3)

    # 3. Create the Agent
    return create_sql_agent(
        llm=agent_llm,
        toolkit=toolkit,
        verbose=True,
        agent_type="openai-tools",  # Groq supports this architecture well
        system_message=_SYSTEM_MESSAGE,
        max_iterations=5,  # Prevents infinite loops if the model gets confused
        # capture the SQL for your frontend!
        agent_executor_kwargs={"return_intermediate_steps": True}
    )
