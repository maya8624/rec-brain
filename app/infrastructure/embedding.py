from typing import Sequence

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.core.config import settings


class EmbeddingService:
    """
    Responsible for generating embeddings from text.
    """

    def __init__(self) -> None:
        self._model = HuggingFaceEmbedding(
            model_name=settings.EMBEDDING_MODEL
        )

    @property
    def model(self) -> HuggingFaceEmbedding:
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        """
        if not text.strip():
            raise ValueError("Cannot embed empty text")

        return self._model.get_text_embedding(text)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        """
        if not texts:
            return []

        return self._model.get_text_embedding_batch(list(texts))
