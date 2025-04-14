from src.llm_client import get_openai, ALL_MODEL
from pathlib import Path
import click
import json
from typing import TypedDict, Callable
from loguru import logger
from src.db.factory import EmbeddingFactory, EmbeddingType, VectorDBFactory, VectorDBType
from src.db.factory import BaseEmbedding, BaseVectorDB
import tqdm
from typing import List
class SpecData(TypedDict):
    summary: str
    python_code: str
    api: List[str]

def get_api_docs(apis:List[str],embedding:BaseEmbedding, vector_db:BaseVectorDB) -> List[str]:
    results:List[str] = []
    for api in apis:
        prompt = f"""documentation for {api} defined by sphinx. using __DOC__ to refer to the documentation.
        """
        query_embedding = embedding.embed_query(prompt)
        results.extend(vector_db.query_by_emb(query_embedding, n_results=2).documents) # type: ignore
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
input_tensor = torch.randn(2, 4, 32, 32).cuda()  # Different input dimensions, use cuda

# Forward pass
output = model(input_tensor)
```
"""
@click.command()
@click.option("--input-dir", type=str, default="spec-2-new", required=True)
@click.option("--output-dir", type=str, default="code-3-new", required=True)
@click.option("--model", type=click.Choice(ALL_MODEL), default="qwen/qwen-2.5-coder-32b-instruct", help="Model to use for generate prompt")
@click.option("--gen-num", type=int, default=5, help="Number of test to generate")
@click.option("--temperature", type=float, default=0.2, help="Temperature for the model")
@click.option("--use-vllm", is_flag=True, default=False,help="whether to use the local vllm server")
def main(input_dir: str, output_dir: str, model: str, gen_num: int, temperature: float, use_vllm: bool):
    input_p:Path = Path(input_dir)
    output_p:Path = Path(output_dir)
    output_p.mkdir(parents=True, exist_ok=True)
    client = get_openai(model,  use_vllm)
    all_specs = list(input_p.glob("*.json"))
    logger.info(f"Generating {gen_num} tests for {len(all_specs)} specs")
    for spec_p in tqdm.tqdm(all_specs):
        func_dir = output_p / spec_p.stem
        func_dir.mkdir(parents=True, exist_ok=True)
        if len(list(func_dir.glob("*.py"))) >= gen_num:
            logger.info(f"Skipping {spec_p} because it already has {gen_num} tests")
            continue
        else:
            gen_num = gen_num - len(list(func_dir.glob("*.py")))
        with open(spec_p, "r") as f:
            spec: SpecData = json.load(f)
        summary = spec["summary"]
        python_code = spec["python_code"]
        api:List[str] = spec["api"]
        embedding = EmbeddingFactory.create_embedding(EmbeddingType.HUGGINGFACE)
        vector_db = VectorDBFactory.create_source_vector_db(vector_db_type=VectorDBType.CHROMA)
        api_docs:List[str] = get_api_docs(api, embedding, vector_db)
        USER_PROMPT = f"""
        Here is the summary of your test goal:
        {summary}
        Here is the python code:
        {python_code}
        Here is the api documentation:
        {"\n".join([f"```doc\n{api_doc}\n```" for api_doc in api_docs])}
        """
        prompt = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT}
            ],
            n=gen_num,
            temperature=temperature
        )
        logger.info(f"Generated {len(prompt.choices)} tests for {spec_p}")
        for idx, choice in enumerate(prompt.choices):
            out_name:str = f"{spec_p.stem}-{idx}.py"
            code:str = extract_python_code(choice.message.content) # type: ignore
            with open(func_dir / out_name, "w") as f:
                f.write(code) # type: ignore
    
if __name__ == "__main__":
    main()
