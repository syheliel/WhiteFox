from typing import List
import numpy as np
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModel
import torch
from .base import BaseEmbedding


class HuggingFaceEmbedding(BaseEmbedding):
    """HuggingFace embedding implementation."""
    tokenizer: AutoTokenizer
    model: AutoModel
    def __init__(self, model_name: str = "Salesforce/codet5p-110m-embedding"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) # type: ignore
        self.model = AutoModel.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.model = self.model.cuda() # type: ignore

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(
                text, padding=True, truncation=True, return_tensors="pt" # type: ignore
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs) # type: ignore
            # Use CLS token embedding
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def embed_query(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt" # type: ignore
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs) # type: ignore
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embedding[0]
