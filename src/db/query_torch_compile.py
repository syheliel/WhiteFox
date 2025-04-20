#!/usr/bin/env python3
"""
Script to query the vector database for code snippets related to torch.compile.
This script retrieves the 10 most relevant code snippets from the vector database
that are related to torch.compile functionality.
"""

import click
from loguru import logger
from src.db.factory import EmbeddingFactory, EmbeddingType, ChromaVectorDB
from rich.console import Console
from typing import List

console = Console()

def format_code_snippet(source: str, content: str) -> str:
    """Format a code snippet with its source for display."""
    return f"```python\n# Source: {source}\n{content}\n```"

@click.command()
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
    "--query",
    type=str,
    default="torch.compile implementation and usage examples",
    help="Query to search for in the vector database",
)
@click.option(
    "--num-results",
    type=int,
    default=10,
    help="Number of results to return",
)
def main(embedding_type: str, model_name: str, query: str, num_results: int):
    """
    Query the vector database for code snippets related to torch.compile.
    """
    logger.info("hello")
    # Create embedding instance
    embedding = EmbeddingFactory.create_embedding(
        EmbeddingType(embedding_type), model_name
    )
    
    # Create vector database instance
    vector_db = ChromaVectorDB(embedding)
    
    # Generate query embedding
    query_embedding = embedding.embed_query(query)
    
    # Query the vector database
    logger.info(f"Querying vector database with: '{query}'")
    results = vector_db.query_by_emb(query_embedding, n_results=num_results)
    docs:List[str] = results["documents"][0] # type: ignore
    for doc in docs: # type: ignore
        print(doc) # type: ignore
    

if __name__ == "__main__":
    main() 