from typing import List
import numpy as np
from langchain_voyageai import VoyageAIEmbeddings
from .base import BaseEmbedding


class VoyageEmbedding(BaseEmbedding):
    """VoyageAI embedding implementation."""

    def __init__(self, model: str = "voyage-code-3", batch_size: int = 20):
        self.embeddings = VoyageAIEmbeddings(model=model, batch_size=batch_size)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return np.array(self.embeddings.embed_documents(texts), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.array(self.embeddings.embed_query(text), dtype=np.float32)
