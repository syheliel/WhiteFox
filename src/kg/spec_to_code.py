from src.conf import TORCH_BASE
from src.llm_client import get_openai, ALL_MODEL
from pathlib import Path
import click
import json
from typing import TypedDict
from loguru import logger
import tqdm
class SpecData(TypedDict):
    summary: str
    python_code: str

extract_python_code = lambda text: text.split("```python")[1].split("```")[0]
SYSTEM_PROMPT = """You are a pytorch source code test generator. I will give you a summary of your test goal and a python code. Please generate a runnable python code that:
1. contains a class that inherit from nn.Module and define forward function. You can use the example code as reference.
2. prepare the input tensor and output tensor for the forward function. the input tensor's shape and dtype should pass the argument check of the forward function.
Here is an example, please output the code as the example format and with no other text.
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
input_tensor = torch.randn(2, 4, 32, 32)  # Different input dimensions

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
def main(input_dir: str, output_dir: str, model: str, gen_num: int, temperature: float):
    input_p:Path = Path(input_dir)
    output_p:Path = Path(output_dir)
    output_p.mkdir(parents=True, exist_ok=True)
    client = get_openai(model)
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
        USER_PROMPT = f"""
        Here is the summary of your test goal:
        {summary}
        Here is the python code:
        {python_code}
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
            out_name = f"{spec_p.stem}-{idx}.py"
            code = extract_python_code(choice.message.content) # type: ignore
            with open(func_dir / out_name, "w") as f:
                f.write(code)
    
if __name__ == "__main__":
    main()