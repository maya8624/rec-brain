from langchain_groq import ChatGroq
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from src.infrastructure.database import get_db_wrapper  # Using your refined DB logic
from src.config import settings


def create_real_estate_sql_agent():
    # 1. Initialize Groq with Llama 3.1 8B
    # temperature=0 is non-negotiable for SQL generation to avoid hallucinations.
    llm = ChatGroq(
        model=settings.MODEL_NAME,
        temperature=0,
        groq_api_key=settings.GROQ_API_KEY
    ).with_retry(stop_after_attempt=3)

    db = get_db_wrapper()
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # 2. Craft a High-Precision System Message
    system_message = (
        "You are a Real Estate SQL Expert. Follow these rules strictly:\n"
        "1. ALWAYS use the 'sql_db_list_tables' tool first.\n"
        "2. NEVER use 'SELECT *'. Query specific columns only.\n"
        "3. ALWAYS add 'LIMIT 10'.\n"
        "4. If you cannot find the answer, say 'I couldn't find a direct match in our listings.'\n"
        "5. Do NOT mention table names or SQL technicalities to the customer."
    )

    # 3. Create the Agent
    return create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type="openai-tools",  # Groq supports this architecture well
        system_message=system_message,
        max_iterations=5,  # Prevents infinite loops if the model gets confused
        # capture the SQL for your frontend!
        agent_executor_kwargs={"return_intermediate_steps": True}
    )
