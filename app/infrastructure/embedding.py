from typing import Sequence

from llama_index.embeddings.openai import OpenAIEmbedding

from app.core.config import settings


class EmbeddingService:
    """
    Responsible for generating embeddings from text.
    """

    def __init__(self) -> None:
        self._model = OpenAIEmbedding(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY,
        )

    @property
    def model(self) -> OpenAIEmbedding:
        return self._model

    def embed_text(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Cannot embed empty text")

        return self._model.get_text_embedding(text)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        return self._model.get_text_embedding_batch(list(texts))
