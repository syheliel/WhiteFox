from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseEmbedding(ABC):
    """Base class for all embedding implementations."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed a list of documents.

        Args:
            texts: List of text documents to embed

        Returns:
            numpy.ndarray: Array of embeddings
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text.

        Args:
            text: Text to embed

        Returns:
            numpy.ndarray: Query embedding
        """
        pass
