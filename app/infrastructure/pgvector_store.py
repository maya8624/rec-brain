from llama_index.vector_stores.postgres import PGVectorStore

from app.core.config import settings


class PgVectorStoreService:
    """
    Creates PGVectorStore using POSTGRES_URL.
    Derives the asyncpg connection string from the same URL.
    """

    def create_vector_store(self) -> PGVectorStore:
        conn_str = str(settings.POSTGRES_URL)

        async_conn_str = (
            conn_str
            .replace("postgresql://", "postgresql+asyncpg://")
            .replace("postgresql+psycopg2://", "postgresql+asyncpg://")
            .replace("sslmode=require", "ssl=require")
        )

        return PGVectorStore.from_params(
            table_name=settings.VECTOR_TABLE,
            embed_dim=settings.EMBEDDING_DIM,
            perform_setup=True,
            connection_string=conn_str,
            async_connection_string=async_conn_str,
        )
