import time
import numpy as np
import argparse
from typing import List
from .db.hg_emb import HuggingFaceEmbedding


def benchmark_embed_query(embedding: HuggingFaceEmbedding, text: str, n_runs: int = 10) -> float:
    """Benchmark the embed_query method."""
    start_time = time.time()
    for _ in range(n_runs):
        embedding.embed_query(text)
    total_time = time.time() - start_time
    avg_time = total_time / n_runs
    return avg_time


def benchmark_embed_documents(embedding: HuggingFaceEmbedding, texts: List[str], n_runs: int = 10) -> float:
    """Benchmark the embed_documents method."""
    start_time = time.time()
    for _ in range(n_runs):
        embedding.embed_documents(texts)
    total_time = time.time() - start_time
    avg_time = total_time / n_runs
    return avg_time


def main():
    parser = argparse.ArgumentParser(description="Benchmark HuggingFaceEmbedding")
    parser.add_argument("--model", type=str, default="./llm_model/iic/gte_Qwen2-1.5B-instruct",
                        help="Path to the HuggingFace model")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to run the model on")
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of runs for averaging performance")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Batch size for embed_documents tests")
    parser.add_argument("--text-length", type=int, default=100,
                        help="Length of randomly generated text")
    args = parser.parse_args()

    print(f"Initializing HuggingFaceEmbedding with model: {args.model} on device: {args.device}")
    embedding = HuggingFaceEmbedding(model_name=args.model, device=args.device)
    
    # Generate random text for testing
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    random_text = "".join(np.random.choice(list(alphabet), size=args.text_length))
    random_texts = [random_text] * args.batch_size
    
    # Warmup
    print("Warming up...")
    embedding.embed_query(random_text)
    embedding.embed_documents(random_texts[:2])
    
    # Test single query embedding
    print(f"\nBenchmarking embed_query with text length: {args.text_length}")
    query_time = benchmark_embed_query(embedding, random_text, args.runs)
    print(f"Average time for embed_query: {query_time:.4f} seconds")
    
    # Test document embedding with various batch sizes
    print(f"\nBenchmarking embed_documents with batch size: {args.batch_size}")
    docs_time = benchmark_embed_documents(embedding, random_texts, args.runs)
    print(f"Average time for embed_documents: {docs_time:.4f} seconds")
    print(f"Average time per document: {docs_time / args.batch_size:.4f} seconds")


if __name__ == "__main__":
    main() 