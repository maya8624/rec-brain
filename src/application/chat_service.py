# from tabulate import tabulate
# import re
# import datetime
# import asyncio
import logging
# Enables logging for this file.
# It acts as the bridge between your API rous(routes.py)
# and the infrastructure components (Vector DB, Sql DB, LLM)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
# Its job is to take a raw user question and decide how to answer it using a "hybrid" approach:
# Vector Search: For conversational or fuzzy queries (e.g., "Find me a cozy apartment").
# SQL Fallback: For structured, data-heavy queries (e.g., "How many listings are in New York?").

from src.infrastructure.sql_agent import create_real_estate_sql_agent
# Import LangChain components needed to build a chain that runs natrual language into SQL
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough

from src.domain.models import ChatResponse
from src.infrastructure.llm import get_llm
from src.infrastructure.vector_db import search_query

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


async def _perform_sql_search(question: str) -> dict[str, str] | None:
    # 1. Initialize your high-powered Agent
    agent_executor = create_real_estate_sql_agent()

    print("🏠 Real Estate Agent is Online.")

    # 2. Single call handles: Schema -> Translation -> Execution -> Formatting
    try:
        # We use .ainvoke for async execution
        response = await agent_executor.ainvoke({"input": question})

        print(f"\n[Agent Answer]: {response['output']}")

    except Exception as e:
        print(f"Agent encountered an unrecoverable error: {e}")


async def handle_user_query(question: str) -> ChatResponse:
    """Process user question via vector search with SQL fallback."""
    logger.info("Handling query: %s", question)

    context = None
    source = "general_knowledge"

    # # 1. Try Vector Search
    # vector_result = await _perform_vector_search(question)

    # # return vector_result["answer"]
    # if vector_result:
    #     context = vector_result.get("answer", "")
    #     source = "vector_db"
    # else:
    # 2. SQL Fallback
    sql_result = await _perform_sql_search(question)

    if sql_result:
        context = sql_result.get("answer", "")
        source = "sql_db"

    answer = await generate_answer(
        context=context or "",
        question=question
    )

    return ChatResponse(
        answer=answer,
        source=source if context else "general_knowledge",
    )


# async def _perform_sql_search_manual(question: str) -> dict[str, str] | None:
    """Encapsulated logic for SQL database search."""
    # Defines a LangChain expression language (LCEL) chain.
    # It passes the database schema, user question, and prompt to the LLM,
    # then parses the raw text output int a clean string
    # tables = db.get_table_info()

    # sql_chain = (
    #     RunnablePassthrough.assign(schema=lambda _: tables)
    #     | prompt_template
    #     | llm
    #     | StrOutputParser()
    # )

    # if not sql_chain:
    #     logger.warning("SQL chain not initialized.")
    #     return None

    # try:
    #     logger.info("Falling back to SQL search: %s", question)

    #     # Generate SQL query
    #     generated_sql = await sql_chain.ainvoke({"question": question})
    #     clean_sql = generated_sql.replace(
    #         "```sql", "").replace("```", "").strip()

    #     # Execute SQL query asynchronously
    #     result = await asyncio.to_thread(db.run, clean_sql)

    #     # Format result into readable text
    #     formatted = _format_sql_result(result, question)

    #     return {
    #         "answer": formatted,
    #         "source": "sql_db",
    #         "query": clean_sql
    #     }
    # except Exception as e:
    #     logger.exception(
    #         "SQL search failed for query: %s, error: %s", question, e)
    #     return None


# def _format_sql_result(result: str, question: str = "") -> str:
#     try:
#         if not result.strip():
#             return f"No results found for '{question}'."

#         dates = re.findall(
#             r'datetime\.date\((\d+),\s*(\d+),\s*(\d+)\)', result)
#         times = re.findall(r'datetime\.time\((\d+),\s*(\d+)\)', result)

#         if not dates:
#             return f"Query: {question}\n\nResults:\n{result}"

#         rows = []
#         for i, d in enumerate(dates):
#             date_str = datetime.date(int(d[0]), int(
#                 d[1]), int(d[2])).strftime("%B %d, %Y")
#             time_str = (
#                 datetime.time(int(times[i][0]), int(
#                     times[i][1])).strftime("%I:%M %p")
#                 if i < len(times) else "N/A"
#             )
#             rows.append((date_str, time_str))

#         table = tabulate(rows, headers=["Date", "Time"], tablefmt="simple")

#         return (
#             f"Query: {question}\n\n"
#             f"Results:\n{table}\n\n"
#             f"Total records found: {len(rows)}"
#         )

#     except Exception:
#         return str(result)
