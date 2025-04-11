from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from src.conf import TORCH_BASE
from src.db.factory import EmbeddingFactory, EmbeddingType, VectorDBFactory, VectorDBType
import click
from loguru import logger
from pathlib import Path
from hashlib import md5
from tqdm import tqdm
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np


def get_id(chunk: Document) -> str:
    return chunk.metadata["source"] + md5(chunk.page_content.encode()).hexdigest()


def unique_chunks(chunks: List[Document]) -> List[Document]:
    all_ids = set()
    unique_chunks = []
    for chunk in chunks:
        if get_id(chunk) not in all_ids:
            unique_chunks.append(chunk)
            all_ids.add(get_id(chunk))
    return unique_chunks


def list_source_files(root: Path, extensions: List[str]) -> List[Path]:
    source_file_paths: List[Path] = []
    for ext in extensions:
        source_file_paths.extend(list(root.rglob(f"*.{ext}")))
    logger.info(f"under {root} found {len(source_file_paths)} source files")
    return source_file_paths


def process_chunk_batch(chunks: List[Document], embedding, vector_db, batch_size: int):
    """Process a batch of chunks in parallel."""
    embeddings = embedding.embed_documents([chunk.page_content for chunk in chunks])
    vector_db.add(
        ids=[get_id(chunk) for chunk in chunks],
        documents=[chunk.page_content for chunk in chunks],
        embeddings=embeddings,
        metadatas=[chunk.metadata for chunk in chunks],
    )


@click.command()
@click.option("--batch-size", type=int, default=20, help="Batch size for embedding")
@click.option("--num-workers", type=int, default=4, help="Number of worker threads")
@click.option(
    "--extensions",
    type=list[str],
    default=["py"],
    help="source code extensions that you what to search",
)
@click.option(
    "--embedding-type",
    type=click.Choice([t.value for t in EmbeddingType]),
    default=EmbeddingType.HUGGINGFACE.value,
    help="Type of embedding to use",
)
@click.option(
    "--model-name",
    type=str,
    help="Model name to use for embedding (optional)",
)
@click.option(
    "--vector-db-type",
    type=click.Choice([t.value for t in VectorDBType]),
    default=VectorDBType.CHROMA.value,
    help="Type of vector database to use",
)
def main(batch_size: int, num_workers: int, extensions: list[str], embedding_type: str, model_name: str, vector_db_type: str):
    """
    embed all source files in torch into vector database
    Here is the default setting:
    - Splitter: RecursiveCharacterTextSplitter with chunk_size=1000, chunk_overlap=200
    - embedding model: voyage-code-3 (default)
    """
    # find all cpp files in mlir/lib and mlir/include
    found_source_file = list_source_files(TORCH_BASE / "torch", extensions)

    docs = [
        Document(page_content=x.read_text(), metadata={"source": str(x)})
        for x in found_source_file
    ]

    logger.info(f"Loaded {len(docs)} with extensions {extensions}")
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)
    chunks = unique_chunks(chunks)
    logger.info(f"Split into {len(chunks)} chunks")

    # vectorDB part - use the new VectorDBFactory
    vector_db = VectorDBFactory.create_source_vector_db(vector_db_type=VectorDBType(vector_db_type))
    
    all_items = vector_db.get()
    all_ids = all_items["ids"]
    new_chunks = list(filter(lambda chunk: get_id(chunk) not in all_ids, chunks))
    logger.info(f"Found {len(new_chunks)} new chunks to embed")
    del chunks

    # Create embedding instance
    embedding = EmbeddingFactory.create_embedding(
        EmbeddingType(embedding_type), model_name
    )

    if len(new_chunks) > 0:
        # Process chunks in parallel batches
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i in range(0, len(new_chunks), batch_size):
                batch_chunks = new_chunks[i : min(i + batch_size, len(new_chunks))]
                future = executor.submit(process_chunk_batch, batch_chunks, embedding, vector_db, batch_size)
                futures.append(future)
            
            # Wait for all batches to complete with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")


if __name__ == "__main__":
    main()
