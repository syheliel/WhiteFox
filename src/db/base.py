from abc import ABC, abstractmethod
from typing import List
import numpy as np
from numpy.typing import NDArray


class BaseEmbedding(ABC):
    """Base class for all embedding implementations."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> NDArray[np.float32]:
        """Embed a list of documents.

        Args:
            texts: List of text documents to embed

        Returns:
            numpy.ndarray: Array of embeddings with dtype float32
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embed a single query text.

        Args:
            text: Text to embed

        Returns:
            numpy.ndarray: Query embedding with dtype float32
        """
        pass
