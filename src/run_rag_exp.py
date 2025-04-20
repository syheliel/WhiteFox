from src.llm_client import get_openai, ALL_MODEL
from pathlib import Path
import click
from typing import TypedDict, Callable
from loguru import logger
from src.db.factory import EmbeddingFactory, EmbeddingType, ChromaVectorDB
from src.db.factory import BaseEmbedding
import tqdm
from typing import List
import concurrent.futures
from functools import partial
from openai import OpenAI
class SpecData(TypedDict):
    summary: str
    python_code: str
    api: List[str]

def get_api_docs(apis:List[str],embedding:BaseEmbedding, vector_db:ChromaVectorDB) -> List[str]:
    results:List[str] = []
    for api in apis:
        prompt = f"""documentation for {api} defined by sphinx. using __DOC__ to refer to the documentation.
        """
        query_embedding = embedding.embed_query(prompt)
        docs:List[str] = vector_db.query_by_emb(query_embedding, n_results=2)["documents"][0] # type: ignore
        results.extend(docs) # type: ignore
    return results
extract_python_code: Callable[[str], str] = lambda text: text.split("```python")[1].split("```")[0]
SYSTEM_PROMPT = """You are a pytorch source code test generator. I will give you a summary of your test goal and a python code. Please generate a runnable python code that:
1. nn.Module class:your code should have one classinherits from nn.Module.
2. prepare input: prepare input tensor and output tensor for the forward function.
3. shape check: I will give you the doc for related api that may contain shape constraint. follow them and calculate the legal input shape
```python
import torch
import torch.nn as nn

class NewModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 16, kernel_size=1, stride=2, padding=0)  # Different parameters
        
    def forward(self, x):
        t1 = self.conv(x)
        t2 = t1 * 0.5
        t3 = t1 * 0.7071067811865476
        t4 = torch.erf(t3)
        t5 = t4 + 1
        t6 = t2 * t5
        return t6

# Initialize the model
model = NewModel()

# Generate input tensor
input_tensor = torch.randn(2, 4, 32, 32)

# Forward pass
output = model(input_tensor)
```
"""

def get_usr_prompt(op:str, api_docs:List[str]) -> str:
    return f"""
    Here is the API that you need to test:
    {op}
    Here is the api documentation:
    {"\n".join([f"```doc\n{api_doc}\n```" for api_doc in api_docs])}
    """

def process_single_op(op: str, output_p: Path, client:OpenAI, model: str, gen_num: int, temperature: float, embedding: BaseEmbedding, vector_db: ChromaVectorDB):
    """Process a single operation and generate tests with and without RAG."""
    logger.info(f"Processing operation: {op}")
    exist_num = len(list(output_p.glob(f"{op}-*.py")))
    if exist_num >= gen_num:
        logger.info(f"Skipping {op} because it already has {gen_num} tests")
        return
    else:
        num_to_generate = gen_num - exist_num
        logger.info(f"Generating {num_to_generate} new tests for {op}")
        
    api_docs:List[str] = get_api_docs([op], embedding, vector_db)
    usr_prompt_with_rag:str = get_usr_prompt(op, api_docs)

    # with RAG
    logger.info(f"Generating tests WITH RAG for {op}...")
    prompt = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": usr_prompt_with_rag}
        ],
        n=num_to_generate,
        temperature=temperature
    )
    logger.info(f"Generated {len(prompt.choices)} tests for {op} with RAG")
    for idx, choice in enumerate(prompt.choices):
        out_name:str = f"{op}-{idx}.py"
        code:str = extract_python_code(choice.message.content) # type: ignore
        rag_out_p:Path = output_p / "with_rag" / out_name
        rag_out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(rag_out_p, "w") as f:
            f.write(code) # type: ignore
        logger.debug(f"Saved test {idx+1}/{len(prompt.choices)} with RAG for {op}")
    
    # without RAG
    logger.info(f"Generating tests WITHOUT RAG for {op}...")
    usr_prompt_without_rag:str = get_usr_prompt(op, [])
    prompt = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": usr_prompt_without_rag}
        ],
        n=num_to_generate,
        temperature=temperature
    )
    logger.info(f"Generated {len(prompt.choices)} tests for {op} without RAG")
    for idx, choice in enumerate(prompt.choices):
        out_name:str = f"{op}-{idx}.py"
        code:str = extract_python_code(choice.message.content) # type: ignore
        rag_out_p:Path = output_p / "without_rag" / out_name
        rag_out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(rag_out_p, "w") as f:
            f.write(code) # type: ignore
        logger.debug(f"Saved test {idx+1}/{len(prompt.choices)} without RAG for {op}")
    
    logger.info(f"✅ Completed processing operation: {op}")
    return op

@click.command()
@click.option("--output-dir", type=str, default="rag_exp", required=True)
@click.option("--model", type=click.Choice(ALL_MODEL), default="qwen/qwen-2.5-coder-32b-instruct", help="Model to use for generate prompt")
@click.option("--device", type=str, default="cpu", help="Device to use for generate prompt")
@click.option("--gen-num", type=int, default=10, help="Number of test to generate")
@click.option("--temperature", type=float, default=0.2, help="Temperature for the model")
@click.option("--use-vllm", is_flag=True, default=False,help="whether to use the local vllm server")
@click.option("--max-workers", type=int, default=2, help="Maximum number of parallel workers")
def main(output_dir: str, model: str, device: str, gen_num: int, temperature: float, use_vllm: bool, max_workers: int):
    output_p:Path = Path(output_dir)
    output_p.mkdir(parents=True, exist_ok=True)
    client = get_openai(model, use_vllm)
    logger.info(f"Using model {client.api_key} on {client.base_url}")
    
    # Create embedding and vector_db once
    embedding = EmbeddingFactory.create_embedding(EmbeddingType.HUGGINGFACE, device=device)
    vector_db = ChromaVectorDB(embedding)
    
    with open("./optim-0/oplist.txt", "r") as f:
        op_list:List[str] = f.read().splitlines()
    
    logger.info(f"Starting to process {len(op_list)} operations with {max_workers} workers")
    
    # Create a partial function with fixed arguments
    process_op_partial = partial(
        process_single_op,
        output_p=output_p,
        client=client,
        model=model,
        gen_num=gen_num,
        temperature=temperature,
        embedding=embedding,
        vector_db=vector_db
    )
    
    # Use ThreadPoolExecutor for I/O-bound operations
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all operations to the executor
        future_to_op = {executor.submit(process_op_partial, op): op for op in op_list}
        
        # Process the results as they complete
        completed = 0
        for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_op), total=len(op_list), desc="Processing operations"):
            op = future_to_op[future]
            completed += 1
            try:
                result = future.result()
                if result:
                    logger.info(f"Completed processing {result} ({completed}/{len(op_list)})")
            except Exception as exc:
                logger.error(f"{op} generated an exception: {exc}")
                raise exc
    
    logger.info(f"✅ All operations processed successfully. Results saved in {output_dir}")

if __name__ == "__main__":
    main()