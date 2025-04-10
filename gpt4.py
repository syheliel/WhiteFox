"""
You will need to use your own OpenAI API key to run this script.
"""

import openai
import time
import os
import json
from pathlib import Path
from logger import logger
import tqdm
import click
import concurrent.futures
from typing import List, Dict, Any
from llm_client import get_openai,ALL_MODEL
system_message = "You are a source code analyzer for {}."

def process_msg(msg):
    """Extract code blocks."""
    if "```" not in msg:
        # the whole response message is a python program
        return msg
    code_st = False
    code = ""
    for line in msg.splitlines():
        if code_st:
            if line.strip().startswith("```"):
                # end of code block
                # but there might be more code blocks
                code_st = False
                continue
            code += line + "\n"
        else:
            if line.strip().startswith("```"):
                code_st = True
    return code

def make_LLM_req(client: openai.OpenAI, system_msg: str, user_input: str, model: str, temperature: float, top_p: float, max_tokens: int) -> Dict[str, Any]:
    """Make a single OpenAI API request with retry logic."""
    max_retries = 3
    retry_delay = 3
    
    logger.info(f"Making LLM request to {model} with temperature={temperature}, top_p={top_p}, max_tokens={max_tokens}")
    
    for attempt in range(max_retries):
        try:
            t_start = time.time()
            logger.debug(f"Attempt {attempt+1}/{max_retries} for LLM request")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_input},
                ],
                max_tokens=max_tokens,
                top_p=top_p,
                temperature=temperature,
                n=1,  # We'll handle multiple requests in parallel
                timeout=300,
            )
            g_time = time.time() - t_start
            logger.info(f"LLM request successful. Response time: {g_time:.2f}s")
            return {
                "response": response,
                "time": g_time
            }
        except Exception as e:
            logger.error(f"LLM request failed on attempt {attempt+1}/{max_retries}: {str(e)}")
            if attempt == max_retries - 1:
                logger.error(f"All retry attempts failed for LLM request to {model}")
                raise e
            logger.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    raise Exception("Failed to make OpenAI request")

@click.command()
@click.option('--prompt-dir', type=str, default="prompt/demo", help="Directory containing prompt files")
@click.option('--outdir', type=str, default="chatgpt/zero-shot", help="Output directory for results")
@click.option('--iter', type=int, default=1, help="Number of iterations to run")
@click.option('--temperature', type=float, default=1.0, help="Temperature for LLM generation")
@click.option('--target', type=str, default="PyTorch", help="Target framework for code analysis")
@click.option('--model', type=str, default="gpt-4", help="LLM model to use")
@click.option('--batch-size', type=int, default=1, help="Batch size for parallel processing")
@click.option('--max-workers', type=int, default=4, help="Maximum number of parallel workers")
@click.option('--max-tokens', type=int, default=2048, help="Maximum number of tokens for LLM generation")
def main(prompt_dir, outdir, iter, temperature, target, model, batch_size, max_workers, max_tokens):
    """Process prompts using OpenAI's API and generate code."""
    system_msg = system_message.format(target)

    prompt_dir = Path(prompt_dir)
    opts = {}
    logger.info(f"prompt dir: {prompt_dir}")
    for prompt_file in prompt_dir.iterdir():
        if not prompt_file.is_file():
            continue
        opts[prompt_file.stem] = prompt_file.read_text()

    outdir = Path(outdir)
    iteration = iter
    top_p = 1.0

    for opt_idx, opt in tqdm.tqdm(enumerate(opts)):
        if os.path.exists(os.path.join(outdir, opt, f"{opt}_1.py")):
            logger.info(f"Skipping opt {opt}")
            continue

        code_idx = 0
        ret = {"opt": opt}
        ret["response"] = {}
        os.makedirs(os.path.join(outdir, opt), exist_ok=True)
        user_input = opts[opt]
        prompt_filepath = os.path.join(outdir, opt, f"prompt.txt")
        logger.info(f"the output will be saved in {prompt_filepath}")
        with open(prompt_filepath, "w") as f:
            f.write(user_input)



        for i in range(iteration):
            logger.info(f"Running {opt} for the {i+1}th time")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_request = {executor.submit(
                    make_LLM_req,
                    get_openai(model),
                    system_msg,
                    user_input,
                    model,
                        temperature,
                        top_p,
                        max_tokens
                    )
                    for _ in range(batch_size)
                }
                
                msgs = []
                total_time = 0
                for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_request), 
                                      total=batch_size,
                                      desc=f"Processing batch {i+1}/{iteration}"):
                    try:
                        result = future.result()
                        msgs.append(result["response"].choices[0].message.content)
                        total_time += result["time"]
                    except Exception as e:
                        logger.error(f"Request failed: {str(e)}")
                        continue

            logger.info(f"[{opt_idx+1}/{len(opts)}] {opt} used time: ", total_time)
            
            codes = []
            for msg in msgs:
                code = process_msg(msg)
                codes.append(code)
                code_idx += 1
                py_filepath:Path = outdir / opt / f"{opt}_{code_idx}.py"
                txt_filepath:Path = outdir / opt / f"{opt}_{code_idx}.txt"
                logger.info(f"Writing generated code to {py_filepath}")
                with open(py_filepath, "w") as f:
                    f.write(code if code is not None else "")
                logger.info(f"Writing raw response to {txt_filepath}")
                with open(txt_filepath, "w") as f:
                    f.write(msg if msg is not None else "")

            # Store the results
            ret["response"][i] = {
                "raw": [msg for msg in msgs],
                "code": codes,
                "g_time": total_time
            }

        output_json_path = os.path.join(outdir, "outputs.json")
        logger.info(f"Appending results to {output_json_path}")
        with open(output_json_path, "a") as f:
            f.write(json.dumps(ret, indent=4) + "\n")

if __name__ == "__main__":
    main()
