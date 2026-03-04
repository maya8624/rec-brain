# Imports Python's built-in logging module to record application events.
import logging

import asyncio

# Imports Chroma from langchain_chroma for vector database operations.
from langchain_chroma import Chroma

# Imports HuggingFaceEmbeddings to convert text queries into vector representations.
from langchain_huggingface import HuggingFaceEmbeddings

# Imports the settings object which holds configuration details like paths and URLs.
from src.core.config import settings

# Initializes a logger for this module, using the module name to trace where logs originate.
logger = logging.getLogger(__name__)

# Initializes the embedding function, using the same model as the LLM for consistency.
embeddings = HuggingFaceEmbeddings(
    # The name of the embedding model to use from settings.
    model=settings.EMBEDDING_MODEL,  # sentence-transformers/all-MiniLM-l6-v2
)

# Defines a function to return the initialized Chroma vector store instance


def get_vector_store() -> Chroma:
    """Returns the initialized Chroma vector store."""

    chroma = Chroma(
        persist_directory=settings.CHROMA_PATH,
        embedding_function=embeddings,

        # The name of the collection within the vector database.
        collection_name="real-estate-brain",
    )

    # client = chroma._client
    # collections = client.list_collections()

    # print(f"Total collections: {len(collections)}")
    # for col in collections:
    #     print(f"Collection: {col.name}, Count: {col.count()}")
    print("CHROMA_PATH :", settings.CHROMA_PATH)
    print("Collection  :", chroma._collection.name)
    print("Doc count   :", chroma._collection.count())  # if 0 = empty!

    return chroma


async def search_query(query: str, k: int = 3) -> list[str]:
    """Performs semantic search on ChromaDB."""
    try:
        # Retrieves the vector store instance.
        vector_store = get_vector_store()

        # Performs a similarity search, returning both document contents and scores.
        # Wrapped in asyncio.to_thread because the underlying chroma library is synchronous.
        # results = await asyncio.to_thread(store.similarity_search_with_score, query, k=k)

        # filtered = [
        #     doc.page_content
        #     for doc, score in results
        #     if score <= settings.SIMILARITY_THRESHOLD
        # ]

        docs = await vector_store.amax_marginal_relevance_search(
            query,
            k,   # top k
            5,  # fetch_k
            0.5  # lambda_mult
        )

        result = [doc.page_content for doc in docs]

        # Combine chunks into one context string
        context = "\n\n".join(result)

        return {
            "answer": context,
            "source": "vector_db",
        }

        # retriever = store.as_retriever(
        #     search_type="mmr",
        #     search_kwargs={  # extra parameters that control how the retriever searches"
        #         "k": k,
        #         "fetch_k": 20,  # initial pool size: 20 most similar chunks and choose 4
        #         "lambda_mult": 0.5  # optional, improves diversity balance
        #     })
        # results = await retriever.ainvoke(query)

        # Returns the list of matching documents and their similarity scores.
        # return [doc.page_content for doc in docs]
    except Exception as ex:
        # Logs an error if the vector search operation fails.
        logger.exception(
            "Vector search failed for query: %s, error: %s", query, ex)

    # Returns an empty list so the service layer can fallback to SQL grac
    return []
