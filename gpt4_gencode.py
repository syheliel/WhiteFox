import argparse
from logger import logger
import time
import os
from pathlib import Path
from pprint import pprint
import sys
from typing import List, TypedDict
import concurrent.futures
import re
import click

# It's recommended to install the openai library: pip install openai
from llm_client import get_openai, ALL_MODEL
from openai import OpenAI

class PromptParams(TypedDict):
    prompt: str
    num: int
    max_tokens: int
    model: str
    temperature: float
    top_p: float
    filename_base: str
    output_dir: Path

def extract_python_code(text: str) -> str:
    """Extract Python code blocks from markdown text."""
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n".join(matches)
    return text

def process_prompt(client: OpenAI, params: PromptParams) -> None:
    """Process a single prompt and save the results."""
    st_time = time.time()
    try:
        response = client.chat.completions.create(
            model=params["model"],
            messages=[{"role": "user", "content": params["prompt"]}],
            n=params["num"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            max_tokens=params["max_tokens"],
        )

        used_time = time.time() - st_time
        logger.info(f"Time taken for {params['filename_base']}: {used_time:.2f} seconds")

        output_file_dir = params["output_dir"] / params["filename_base"]
        output_file_dir.mkdir(exist_ok=True, parents=True)

        for r, choice in enumerate(response.choices):
            generated_text = choice.message.content
            if choice.finish_reason == "length":
                logger.warning(f"Generation for {params['filename_base']}-{r} potentially truncated due to max_tokens limit.")
            elif choice.finish_reason != "stop":
                logger.warning(f"Generation for {params['filename_base']}-{r} finished due to: {choice.finish_reason}")

            # Extract Python code blocks
            extracted_code = extract_python_code(generated_text or "")

            output_path = output_file_dir / f"{params['filename_base']}-{r}.py"
            try:
                output_path.write_text(extracted_code, encoding='utf-8')
            except Exception as e:
                logger.error(f"Failed to write output file {output_path}: {e}")

    except Exception as e:
        logger.error(f"An unexpected error occurred for prompt {params['filename_base']}: {e}")

@click.command()
@click.option('--prompt-dir', type=str, default='prompt-3', help='Directory containing prompt files (.txt)')
@click.option('--output-dir', type=str, default='gencode-4', help='Directory to save generated outputs')
@click.option('-n', '--num', type=int, default=1, help='Number of samples to generate per prompt.')
@click.option('--max-tokens', type=int, default=4096, help='Maximum number of tokens to generate.')
@click.option('--model', type=click.Choice(ALL_MODEL), default='gpt-4', help='OpenAI model to use.')
@click.option('--temperature', type=float, default=0.7, help='Sampling temperature.')
@click.option('--top-p', type=float, default=1.0, help='Nucleus sampling p.')
@click.option('--max-workers', type=int, default=4, help='Maximum number of worker threads.')
def main(prompt_dir, output_dir, num, max_tokens, model, temperature, top_p, max_workers):
    """Generate code using OpenAI GPT models."""
    
    client = get_openai(model)

    prompt_dir = Path(prompt_dir)
    prompt_dir.mkdir(exist_ok=True, parents=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    prompt_content:List[str] = []
    filenames:List[str] = []
    for prompt_file in sorted(prompt_dir.glob("*.txt")):
        try:
            with open(prompt_file) as f:
                prompt_content.append(f.read())
            filenames.append(prompt_file.stem)
        except Exception as e:
            logger.warning(f"Could not read prompt file {prompt_file}: {e}")

    num_prompts = len(prompt_content)
    logger.info(f"Number of prompts loaded: {num_prompts}")

    total_st_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, (prompt, filename_base) in enumerate(zip(prompt_content, filenames)):
            # Check if this prompt has already been generated
            output_file_dir = output_dir / filename_base
            if output_file_dir.exists():
                # Check if all expected output files exist
                expected_files = [f"{filename_base}-{j}.py" for j in range(num)]
                existing_files = [f.name for f in output_file_dir.glob("*.py")]
                
                if all(f in existing_files for f in expected_files):
                    logger.info(f"Skipping {filename_base} - already generated with {num} outputs")
                    continue
                else:
                    logger.info(f"Regenerating {filename_base} - missing some outputs (found {len(existing_files)}/{num})")
            else:
                logger.info(f"Generating new outputs for {filename_base}")
            
            promptParams:PromptParams = {
                "prompt": prompt,
                "num": num,
                "max_tokens": max_tokens,
                "model": model,
                "temperature": temperature,
                "top_p": top_p,
                "filename_base": filename_base,
                "output_dir": output_dir
            }
            future = executor.submit(process_prompt, client, promptParams)
            futures.append(future)
        
        concurrent.futures.wait(futures)

    total_used_time = time.time() - total_st_time
    logger.info(f"Total generation finished. Total time: {total_used_time:.2f} seconds")

if __name__ == "__main__":
    main()