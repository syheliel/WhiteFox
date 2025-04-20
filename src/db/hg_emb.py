from typing import List, Optional
from loguru import logger
from numpy.typing import NDArray
import numpy as np
from sentence_transformers import SentenceTransformer
from .base import BaseEmbedding


class HuggingFaceEmbedding(BaseEmbedding):
    """HuggingFace embedding implementation using SentenceTransformer."""
    _model: Optional[SentenceTransformer] = None
    _device: Optional[str] = None # Store device at class level

    def __init__(self, model_name: str = "./llm_model/iic/gte_Qwen2-1.5B-instruct", device: str = "cpu"):
        # Initialize model only once using class attributes
        if HuggingFaceEmbedding._model is None:
            logger.info(f"Initializing SentenceTransformer with model: {model_name} on device: {device}")
            # SentenceTransformer handles moving the model to the specified device internally
            try:
                HuggingFaceEmbedding._model = SentenceTransformer(model_name, device=device)
                HuggingFaceEmbedding._device = device
                logger.info(f"SentenceTransformer model initialized and on device: {HuggingFaceEmbedding._device}")
            except Exception as e:
                logger.error(f"Failed to initialize SentenceTransformer model: {e}")
                # Optionally re-raise or handle the error appropriately
                raise
        # Ensure the instance knows the device being used by the class model
        self.device = HuggingFaceEmbedding._device

    def embed_documents(self, texts: List[str]) -> NDArray[np.float32]:
        if HuggingFaceEmbedding._model is None or HuggingFaceEmbedding._device is None:
            raise RuntimeError("Model not initialized. Call __init__ first.")
        logger.debug(f"Embedding {len(texts)} documents using device: {self.device}")
        embeddings: NDArray[np.float32] = HuggingFaceEmbedding._model.encode( # type: ignore
            texts,
            convert_to_numpy=True,
            device=self.device, # Pass device explicitly, though model is already on it
            show_progress_bar=False # Optional: disable progress bar if noisy
        )
        logger.debug(f"Generated embeddings of shape: {embeddings.shape}")
        return embeddings

    def embed_query(self, text: str) -> NDArray[np.float32]:
        if HuggingFaceEmbedding._model is None or HuggingFaceEmbedding._device is None:
            raise RuntimeError("Model not initialized. Call __init__ first.")
        logger.debug(f"Embedding query using device: {self.device}")
        embedding: NDArray[np.float32] = HuggingFaceEmbedding._model.encode( # type: ignore
            text,
            convert_to_numpy=True,
            device=self.device, # Pass device explicitly
            show_progress_bar=False
        )
        logger.debug(f"Generated query embedding of shape: {embedding.shape}")
        # SentenceTransformer returns 1D array for single text input
        return embedding
