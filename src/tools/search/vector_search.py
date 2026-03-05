# Imports Python's built-in logging module to record application events.
import logging

# Imports Chroma from langchain_chroma for vector database operations.
from langchain_chroma import Chroma

# Imports HuggingFaceEmbeddings to convert text queries into vector representations.
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

# Imports the settings object which holds configuration details like paths and URLs.
from src.api.schemas.chat import ChatResponse
from src.core.config import settings
from src.infrastructure.llm import get_llm

# Initializes a logger for this module, using the module name to trace where logs originate.
logger = logging.getLogger(__name__)

# Initializes the embedding function, using the same model as the LLM for consistency.
embeddings = HuggingFaceEmbeddings(
    model_name=settings.EMBEDDING_MODEL,  # sentence-transformers/all-MiniLM-l6-v2
)

logger = logging.getLogger(__name__)

# Initializes the Large Language Model connection
llm = get_llm()


async def perform_vector_search(query: str) -> dict[str, str] | None:
    """Encapsulated logic for vector database search using MMR."""

    try:
        # Retrieves the vector store instance.
        vector_store = get_vector_store()

        docs = await vector_store.amax_marginal_relevance_search(
            query,
            3,  # TODO: top k, use settings.TopKResults
            5,  # TODO: fetch_k, use settings.FetchKResults
            0.5  # TODO: lambda_mult, use settings.LambdaMult
        )

        result = [doc.page_content for doc in docs]

        # Combine chunks into one context string
        vector_result = "\n\n".join(result)

        # return vector_result["answer"]
        if vector_result:
            context = vector_result

            answer = await generate_answer(
                context=context or "",
                question=query
            )

            return ChatResponse(
                answer=answer,
                source="vector_db" if context else "general_knowledge",
            )

    except Exception as ex:
        logger.exception(
            "Vector search handler failed for query: %s, error: %s",
            query,
            ex
        )
        return None


def get_vector_store() -> Chroma:
    """Returns the initialized Chroma vector store."""

    chroma = Chroma(
        persist_directory=settings.CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="real-estate-brain",
    )

    # print("CHROMA_PATH :", settings.CHROMA_PATH)
    # print("Collection  :", chroma._collection.name)
    # print("Doc count   :", chroma._collection.count())  # if 0 = empty!

    return chroma


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
