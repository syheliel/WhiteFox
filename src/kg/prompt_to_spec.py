import json
import argparse
from pathlib import Path
from typing import List, Dict, TypedDict, Literal, Union
import numpy as np
import random
import os
from src.llm_client import get_openai, ALL_MODEL
# noqa: F401
from src.conf import TORCH_BASE 
from loguru import logger
import click
from pathlib import Path
import tqdm
import concurrent.futures
import yaml
import threading

class Hint(TypedDict):
    type: str
    target_line: str
    func: str
    comment: str

class FuncLevelPrompt(TypedDict):
    hints: List[Hint]
    

def extract_python_code(prompt: str) -> str:
    python_start = prompt.find('```python')
    python_end = prompt.rfind('```')
    if python_start != -1 and python_end != -1:
        return prompt[python_start+len('```python'):python_end]
    return ""

def extract_summary(prompt: str) -> str:
    summary_start = prompt.find('```summary')
    summary_end = prompt.rfind('```')
    if summary_start != -1 and summary_end != -1:
        return prompt[summary_start+len('```summary'):summary_end]
    return ""

def extract_api(prompt: str) -> str:
    summary_start = prompt.find('```yaml')
    summary_end = prompt.rfind('```')
    if summary_start != -1 and summary_end != -1:
        return prompt[summary_start+len('```yaml'):summary_end]
    return ""

def get_spec_name(module_name:str, func_name:str) -> str:
    return f"{module_name}:{func_name}.json"

def module_to_path(module_name:str, base_path:Path=TORCH_BASE) -> Path:
    module_name = module_name.replace(".", "/") + ".py"
    return base_path / module_name

SYSTEM_PROMPT = r"""
You are an expert in pytorch bug hunting. I will give you a specific source code from pytorch and a list of function that is vulnerable. please:
1. SUMMARY: summarize the function's usage
2. EXAMPLE: give an example of how to trigger the target_line in using normal pytorch code without touch the inner API. the python code must only contain one pytorch module
3. potential API: list the potential api 
Here is your example output:
```summary
The BatchLayernormFusion class handles fusing multiple layer normalization operations in PyTorch graphs. The vulnerable line checks that all epsilon values used in the layer norm operations are equal before fusing them. This is important because:
1. Layer normalization uses epsilon for numerical stability
2. Different epsilon values would produce mathematically different results
3. The fusion assumes consistent epsilon values across operations
4. Missing validation could lead to incorrect fused results if epsilons differ
```
```python
import torch
import torch.nn as nn

class FusedBatchLayerNorm(nn.Module):
    """
    融合 BatchNorm 和 LayerNorm 的模块
    """
    def __init__(self, num_features):
        super(FusedBatchLayerNorm, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.ln = nn.LayerNorm(num_features)

    def forward(self, x):
        x = self.bn(x)
        x = self.ln(x)
        return x
```
```yaml
- nn.BatchNorm1d
- nn.LayerNorm
```
When it's your turn, please output the summary,example code and the pentential api as the example format and with no other text.
"""

# Thread-local storage for LLM clients
thread_local = threading.local()

def get_thread_llm(model):
    """Get or create an LLM client for the current thread"""
    if not hasattr(thread_local, "llm"):
        thread_local.llm = get_openai(model)
    return thread_local.llm

def process_prompt_item(prompt_item, output_p, model):
    """Process a single prompt item and save the results"""
    file_name = prompt_item["file_name"]
    func_name = prompt_item["func_name"]
    func_info = prompt_item["func_info"]
    
    spec_name = get_spec_name(file_name, func_name)
    
    if output_p / spec_name in output_p.iterdir():
        logger.info(f"Skipping {spec_name} because it already exists")
        return
        
    with open(module_to_path(file_name.replace(".json", "")), "r") as f:
        file_source_code:str = f.read()
        
    USER_PROMPT = f"""
    Here is the source code:
    ```python
    {file_source_code}
    ```
    Here is the vulerable function's info:
    ```
    {json.dumps(func_info, indent=4)}
    ```
    Please output the summary,example code and potential API as the example format and with no other text.
    """
    
    llm = get_thread_llm(model)
    prompt = llm.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ]
    )
    summary = extract_summary(prompt.choices[0].message.content) # type: ignore
    python_code = extract_python_code(prompt.choices[0].message.content) # type: ignore
    api = extract_api(prompt.choices[0].message.content) # type: ignore
    res = {
        "summary": summary,
        "python_code": python_code,
        "api": api
    }
    
    # Use a lock to prevent concurrent writes to the same file
    with threading.Lock():
        with open(output_p / spec_name, "w") as f:
            f.write(json.dumps(res, indent=4))
        
        with open(output_p / f"{spec_name}.prompt", "w") as f:
            f.write(USER_PROMPT)
    
    return spec_name

@click.command()
@click.option("--input-dir", type=str, default="./prompt-1-new", required=True)
@click.option("--output-dir", type=str, default="spec-2-new", required=True)
@click.option("--model", type=click.Choice(ALL_MODEL), default="deepseek-v3-250324", help="Model to use for generate prompt")
@click.option("--max-workers", type=int, default=4, help="Maximum number of worker threads")
def main(input_dir: str, output_dir: str, model: str, max_workers: int):
    input_p:Path = Path(input_dir)
    output_p:Path = Path(output_dir)
    output_p.mkdir(parents=True, exist_ok=True)
    file_level_prompts: List[Dict[str, FuncLevelPrompt]] = []
    file_level_names: List[str] = []
    for prompt_p in input_p.glob("*.json"):
        try:
            with open(prompt_p, "r") as f:
                prompt = json.load(f)
                file_level_prompts.append(prompt)
                file_level_names.append(prompt_p.stem)
        except Exception as e:
            logger.error(f"Error loading prompt {prompt_p}: {e}")
            continue
    
    logger.info(f"Found {len(file_level_prompts)} file-level prompts")

    # Create a unified list of all file_level_prompts and func_level_prompts combinations
    unified_prompts = []
    for file_idx, file_prompt in enumerate(file_level_prompts):
        file_name = file_level_names[file_idx]
        for func_name, func_info in file_prompt.items():
            unified_prompts.append({
                "file_name": file_name,
                "func_name": func_name,
                "func_info": func_info
            })
    
    logger.info(f"Created unified list with {len(unified_prompts)} prompt combinations")

    # Process prompts using a thread pool
    logger.info(f"Processing prompts with {max_workers} worker threads")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        future_to_prompt = {
            executor.submit(process_prompt_item, prompt_item, output_p, model): prompt_item 
            for prompt_item in unified_prompts
        }
        
        # Process results as they complete
        for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_prompt), total=len(unified_prompts)):
            prompt_item = future_to_prompt[future]
            try:
                spec_name = future.result()
                if spec_name:
                    logger.info(f"Completed processing {spec_name}")
            except Exception as e:
                logger.error(f"Error processing {prompt_item['file_name']}:{prompt_item['func_name']}: {e}")

if __name__ == "__main__":
    main()

    