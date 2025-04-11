from enum import Enum
from typing import Optional, TypedDict
from .base import BaseEmbedding
from .voyage_emb import VoyageEmbedding
from .hg_emb import HuggingFaceEmbedding
from abc import ABC, abstractmethod
import chromadb
from typing import List
import numpy as np
from src.conf import CHROMADB_COLLECTION, CHROMADB_PATH
from pathlib import Path
from hashlib import md5
from chromadb.api.types import GetResult, QueryResult
from loguru import logger
from concurrent.futures import ProcessPoolExecutor, as_completed

class EmbeddingType(Enum):
    VOYAGE = "voyage"
    HUGGINGFACE = "huggingface"
    
class VectorDBType(Enum):
    CHROMA = "chroma"

class BaseVectorDB(ABC):
    embedding: BaseEmbedding
    """Base class for vector database implementations."""
    
    @abstractmethod
    def add(self, ids:List[str], documents:List[str], embeddings:np.ndarray, metadatas:Optional[List[dict]]=None):
        """Add documents to the vector database."""
        pass
    
    @abstractmethod
    def add_by_files(self, file_path:List[Path]):
        """Add documents to the vector database by file path."""
        pass
    
    @abstractmethod
    def get(self, ids=None) -> GetResult:
        """Get documents from the vector database."""
        pass
    
    @abstractmethod
    def query_by_emb(self, query_embeddings, n_results=10) -> QueryResult:
        """Query the vector database."""
        pass

class Metadata(TypedDict):
    source: str

class ChromaVectorDB(BaseVectorDB):
    """ChromaDB implementation of vector database."""
    
    def __init__(self, collection_name: str = CHROMADB_COLLECTION, db_path=CHROMADB_PATH, embedding_type:EmbeddingType=EmbeddingType.VOYAGE):
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedding = EmbeddingFactory.create_embedding(embedding_type)
    
    def add(self, ids:List[str], documents:List[str], embeddings:np.ndarray, metadatas:Optional[List[Metadata]]=None): # type: ignore
        """Add documents to the vector database."""
        logger.info(f"Adding {len(documents)} documents to the {self.collection.name} vector database.")
        return self.collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas) # type: ignore
    
    def add_by_files(self, file_path:List[Path]):
        """Add documents to the vector database by file path."""
        texts = [file_path.read_text() for file_path in file_path]
        embeddings = self.embedding.embed_documents(texts)
        ids = [self.calcu_id(file_path) for file_path in file_path]
        metadatas = [Metadata(source=str(file_path)) for file_path in file_path]
        self.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        
    def add_by_file_batch(self, file_path: List[Path]):
        """Add documents to the vector database by file path in parallel batches."""
        with ProcessPoolExecutor(max_workers=8) as executor:
            # Process files in parallel to get texts and embeddings
            future_to_file = {
                executor.submit(lambda p: (p, p.read_text()), path): path 
                for path in file_path
            }
            
            texts = []
            paths = []
            for future in as_completed(future_to_file):
                try:
                    path, text = future.result()
                    texts.append(text)
                    paths.append(path)
                except Exception as e:
                    logger.error(f"Failed to process {future_to_file[future]}: {e}")
                    continue
                    
            # Get embeddings in parallel batches
            embeddings = self.embedding.embed_documents(texts)
            
            # Generate metadata and IDs
            ids = [self.calcu_id(p) for p in paths]
            metadatas = [Metadata(source=str(p)) for p in paths]
            
            # Add to vector DB
            self.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    
    def get(self, ids=None) -> GetResult:
        """Get documents from the vector database."""
        result = self.collection.get(ids=ids)
        # Handle empty collection case
        if not result or "ids" not in result:
            return {"ids": [], "documents": [], "metadatas": [], "embeddings": []} # type: ignore
        return result
    
    def calcu_id(self, file_path:Path) -> str:
        """Calculate the id of the document."""
        return file_path.name + md5(file_path.read_text().encode()).hexdigest()
    
    def query_by_emb(self, query_embeddings, n_results=10):
        """Query the vector database."""
        return self.collection.query(query_embeddings=query_embeddings, n_results=n_results)
    
    def query_by_text(self, query_text, n_results=10):
        """Query the vector database by text."""
        return self.collection.query(query_texts=[query_text], n_results=n_results)


class EmbeddingFactory:
    """Factory class for creating embedding instances."""

    @staticmethod
    def create_embedding(
        embedding_type: EmbeddingType, model_name: Optional[str] = None
    ) -> BaseEmbedding:
        """Create an embedding instance based on the specified type.

        Args:
            embedding_type: Type of embedding to create
            model_name: Optional model name to use (if applicable)

        Returns:
            BaseEmbedding: An instance of the specified embedding type
        """
        if embedding_type == EmbeddingType.VOYAGE:
            return VoyageEmbedding(model=model_name or "voyage-code-3")
        elif embedding_type == EmbeddingType.HUGGINGFACE:
            return HuggingFaceEmbedding(    
                model_name=model_name or "Salesforce/codet5p-110m-embedding"
            )
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")

class VectorDBFactory:
    @staticmethod
    def create_vector_db(vector_db_type: VectorDBType = VectorDBType.CHROMA, collection_name: str = CHROMADB_COLLECTION) -> BaseVectorDB:
        if vector_db_type == VectorDBType.CHROMA:
            return ChromaVectorDB(collection_name)
        else:
            raise ValueError(f"Unsupported vector database type: {vector_db_type}")
    
    @staticmethod
    def create_source_vector_db(vector_db_type: VectorDBType = VectorDBType.CHROMA) -> BaseVectorDB:
        return VectorDBFactory.create_vector_db(vector_db_type=vector_db_type, collection_name=CHROMADB_COLLECTION)
