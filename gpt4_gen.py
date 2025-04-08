import argparse
from logger import logger
import time
import os
from pathlib import Path
from pprint import pprint
import sys
from typing import List
import concurrent.futures
import re

# It's recommended to install the openai library: pip install openai
from openai import OpenAI, APIError

# Consider adjusting these stop sequences for GPT-4 if needed
EOF_STRINGS = [
    "<|endoftext|>",
    '"""',
    "'''",
    "```", # Added common markdown code block end
]

def extract_python_code(text: str) -> str:
    """Extract Python code blocks from markdown text."""
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n".join(matches)
    return text

def process_prompt(client: OpenAI, prompt: str, args, filename_base: str, output_dir: Path) -> None:
    """Process a single prompt and save the results."""
    st_time = time.time()
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            n=args.num,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )

        used_time = time.time() - st_time
        logger.info(f"Time taken for {filename_base}: {used_time:.2f} seconds")

        output_file_dir = output_dir / filename_base
        output_file_dir.mkdir(exist_ok=True, parents=True)

        for r, choice in enumerate(response.choices):
            generated_text = choice.message.content
            if choice.finish_reason == "length":
                logger.warning(f"Generation for {filename_base}-{r} potentially truncated due to max_tokens limit.")
            elif choice.finish_reason != "stop":
                logger.warning(f"Generation for {filename_base}-{r} finished due to: {choice.finish_reason}")

            # Extract Python code blocks
            extracted_code = extract_python_code(generated_text or "")

            output_path = output_file_dir / f"{filename_base}-{r}.py"
            try:
                output_path.write_text(extracted_code, encoding='utf-8')
            except Exception as e:
                logger.error(f"Failed to write output file {output_path}: {e}")

    except Exception as e:
        logger.error(f"An unexpected error occurred for prompt {filename_base}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate code using OpenAI GPT models.")
    parser.add_argument("--prompt-dir", type=str, default="prompts", help="Directory containing prompt files (.txt)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated-outputs-gpt4",
        help="Directory to save generated outputs"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key. If not provided, tries to read from OPENAI_API_KEY environment variable.",
    )
    parser.add_argument("-n", "--num", type=int, default=1, help="Number of samples to generate per prompt.")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum number of tokens to generate.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model to use.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling p.")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of parallel workers.")

    args = parser.parse_args()
    pprint(args)

    logger.info(f"Starting generation with args: {args}")

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    prompt_dir = Path(args.prompt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    prompt_content:List[str] = []
    filenames:List[str] = []
    if not prompt_dir.is_dir():
        logger.error(f"Prompt directory not found: {prompt_dir}")
        sys.exit(1)

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

    # Process prompts in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for i, (prompt, filename_base) in enumerate(zip(prompt_content, filenames)):
            future = executor.submit(
                process_prompt,
                client,
                prompt,
                args,
                filename_base,
                output_dir
            )
            futures.append(future)
        
        # Wait for all tasks to complete
        concurrent.futures.wait(futures)

    total_used_time = time.time() - total_st_time
    logger.info(f"Total generation finished. Total time: {total_used_time:.2f} seconds")

if __name__ == "__main__":
    main()