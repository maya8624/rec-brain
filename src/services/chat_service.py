import asyncio
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from src.domain.models import ChatResponse

from src.infrastructure.sql_agent import create_real_estate_sql_agent
from src.infrastructure.llm import get_llm
from src.infrastructure.vector_db import search_query
from src.infrastructure.sql_inspector import track_sql_steps

# Sets up the logger instance
logger = logging.getLogger(__name__)

# Initializes the Large Language Model connection
llm = get_llm()

# Defines the instructions for the LLM to convert questions into SQL based on the database schema.
prompt_template = ChatPromptTemplate.from_template(
    """Given the following SQL tables: {schema}
    Answer the question by generating a SQL query only. Do not explain.
    Question: {question}
    SQL Query:"""
)


async def generate_answer(context: str, question: str) -> list[str]:
    messages = [
        SystemMessage(content="""
        You are a helpful assistant with access to two knowledge sources:
        1. Vector Database (documents/files)
        2. SQL Database (structured data like schedules, records, properties)

        Guidelines:
        - Answer using ONLY the provided context below.
        - If the context comes from the SQL database, interpret the structured data (dates, times, records) and answer confidently.
        - If the context comes from the Vector database, use the document content to answer.
        - Give a full, clear answer in complete sentences.
        - If the context does not contain the answer, use your general knowledge and indicate that the answer is not directly from the knowledge sources.
        """),

        HumanMessage(content=f"""
        Context:
        {context}

        Question:
        {question}
        """)
    ]

    response = await llm.ainvoke(messages)
    return response.content


async def _perform_vector_search(question: str) -> dict[str, str] | None:
    """Encapsulated logic for vector database search using MMR."""

    try:
        result = await search_query(question)

        return {
            "answer": result["answer"],
            "source": "vector_db"
        }

    except Exception as ex:
        logger.exception(
            "Vector search handler failed for question: %s, error: %s",
            question,
            ex
        )
        return None


_AGENT_TIMEOUT_SECONDS = 30.0


async def _perform_sql_search(question: str) -> dict[str, str] | None:
    if not question or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    agent_executor = create_real_estate_sql_agent()

    # Schema -> Translation -> Execution -> Formatting
    try:
        response = await asyncio.wait_for(
            agent_executor.ainvoke({"input": question}),
            timeout=_AGENT_TIMEOUT_SECONDS
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
        return None


async def handle_user_query(question: str) -> ChatResponse:
    """Process user question via vector search with SQL fallback."""
    logger.info("Handling query: %s", question)

    # context = None
    # source = "general_knowledge"

    # # 1. Try Vector Search
    # TODO: remove commnets later
    # vector_result = await _perform_vector_search(question)

    # # return vector_result["answer"]
    # if vector_result and vector_result.get("answer"):
    #     context = vector_result.get("answer", "")

    #     answer = await generate_answer(
    #         context=context or "",
    #         question=question
    #     )

    #     return ChatResponse(
    #         answer=answer,
    #         source="vector_db" if context else "general_knowledge",
    #     )
    # else:
    # 2. SQL Fallback
    sql_result = await _perform_sql_search(question)

    if sql_result and sql_result.get("answer"):
        return ChatResponse(
            answer=sql_result["answer"],
            source="sql_db"
        )
