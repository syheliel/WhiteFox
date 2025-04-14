#!/usr/bin/env python3
import sys
import subprocess
import click
from transformers.models.auto.tokenization_auto import AutoTokenizer
from src.llm_client import get_openai,ALL_MODEL
from src.conf import OPTINFO_PATH, TORCH_BASE
from typing import List
def count_tokens(text:str, model_name:str="deepseek-ai/DeepSeek-V3") -> int:
    """
    Count the number of tokens in the given text using the DeepSeek tokenizer.
    
    Args:
        text (str): The text to count tokens for
        model_name (str): The name of the model to use for tokenization
        
    Returns:
        int: The number of tokens in the text
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize the text
    tokens:List[int] = tokenizer.encode(text)
    
    # Return the number of tokens
    return len(tokens)

def run_tree_command(directory: str = ".", max_depth:int = 1) -> str:
    """Run the tree command and return its output as a string."""
    try:
        result = subprocess.run(["tree", directory, "-L", str(max_depth)], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running tree command: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'tree' command not found. Please install it first.", file=sys.stderr)
        sys.exit(1)

SYSTEM_PROMPT = """
You are an expert in pytorch. please analyze the directory structure of the project and give a list of all files that are related to pytorch compile optimization related to torch.compile. Here is an example of the output:
```yaml
torch/_dynamo/:
  - __init__.py
  - backends/:
    - __init__.py
    - common.py
    - cudagraphs.py
    - debugging.py
    - distributed.py
    - inductor.py
    - onnxrt.py
    - registry.py
    - tensorrt.py
    - torchxla.py
    - tvm.py
  - bytecode_analysis.py
  - bytecode_transformation.py
  - compiled_autograd.py
  - config.py
  - convert_frame.py
```
Please only output the yaml content, and all item for the yaml structure. No other text should be output.
"""
@click.command()
@click.argument('directory', default=str(TORCH_BASE/"torch"), type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True))
@click.option('-l', '--depth', type=int, default=3, help='Depth of the tree (default: 3)')
@click.option('-m', '--model', type=str, default="deepseek-v3-250324", help='Model to use for tokenization (default: deepseek-v3-250324)')
def main(directory: str, depth: int, model: str):
    tree_output = run_tree_command(directory, depth)
    print(f"tree output has {count_tokens(tree_output)} tokens")
    llm = get_openai(model)

    USER_PROMPT = f"""
    Here is the directory structure of the project:
    {tree_output}
    Please give a list of all files that are related to pytorch optimization.
    """
    response = llm.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": USER_PROMPT}],
    )
    print(response.choices[0].message.content)
    # Extract the YAML content between ```yaml and ``` markers
    yaml_content = response.choices[0].message.content
    start_marker = "```yaml"
    end_marker = "```"
    
    if yaml_content and start_marker in yaml_content:
        start_idx = yaml_content.find(start_marker) + len(start_marker)
        end_idx = yaml_content.find(end_marker, start_idx)
        if end_idx != -1:
            yaml_content = yaml_content[start_idx:end_idx].strip()
    with open(OPTINFO_PATH, "w") as f:
        f.write(yaml_content or "")

if __name__ == "__main__":
    main()
