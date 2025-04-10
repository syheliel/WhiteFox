"""
You will need to use your own OpenAI API key to run this script.
"""

import argparse
import openai
import time
import os
import json
from pathlib import Path
from logger import logger
import concurrent.futures
from typing import List, Dict, Any

# You need to create a file named "openai.key" and put your API key in it
openai.api_key = Path("openai.key").read_text().strip()
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

MAX_TOKENS = 2048

def make_openai_request(system_msg: str, user_input: str, model: str, temperature: float, top_p: float) -> Dict[str, Any]:
    """Make a single OpenAI API request with retry logic."""
    max_retries = 3
    retry_delay = 3
    
    for attempt in range(max_retries):
        try:
            t_start = time.time()
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_input},
                ],
                max_tokens=MAX_TOKENS,
                top_p=top_p,
                temperature=temperature,
                n=1,  # We'll handle multiple requests in parallel
                timeout=300,
            )
            g_time = time.time() - t_start
            return {
                "response": response,
                "time": g_time
            }
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(retry_delay)
        raise Exception("Failed to make OpenAI request")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-dir", type=str, default="prompt/demo")
    parser.add_argument("--outdir", type=str, default="chatgpt/zero-shot")
    parser.add_argument("--iter", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--prompt-only", action="store_true")
    parser.add_argument("--target", type=str, default="PyTorch")
    parser.add_argument("--model", type=str, default="gpt-4")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of parallel workers")
    args = parser.parse_args()

    system_message = system_message.format(args.target)

    prompt_dir = Path(args.prompt_dir)
    opts = {}
    logger.info(f"prompt dir: {prompt_dir}")
    for prompt_file in prompt_dir.iterdir():
        if not prompt_file.is_file():
            continue
        opts[prompt_file.stem] = prompt_file.read_text()

    outdir = Path(args.outdir)
    iteration = args.iter
    top_p = 1.0
    temperature = args.temperature
    n_batch_size = args.batch_size

    for opt_idx, opt in enumerate(opts):
        if os.path.exists(os.path.join(outdir, opt, f"{opt}_1.py")):
            print("Skipping opt ", opt)
            continue

        code_idx = 0
        ret = {"opt": opt}
        ret["response"] = {}
        os.makedirs(os.path.join(outdir, opt), exist_ok=True)
        user_input = opts[opt]
        prompt_filepath = os.path.join(outdir, opt, f"prompt.txt")
        logger.info(f"Writing prompt to {prompt_filepath}")
        with open(prompt_filepath, "w") as f:
            f.write(user_input)

        if args.prompt_only:
            print(opt_idx)
            continue

        for i in range(iteration):
            # Create a thread pool for parallel requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                # Submit all requests to the thread pool
                future_to_request = {
                    executor.submit(
                        make_openai_request,
                        system_message,
                        user_input,
                        args.model,
                        temperature,
                        top_p
                    ): _ for _ in range(n_batch_size)
                }
                
                # Collect results as they complete
                msgs = []
                total_time = 0
                for future in concurrent.futures.as_completed(future_to_request):
                    try:
                        result = future.result()
                        msgs.append(result["response"].choices[0].message.content)
                        total_time += result["time"]
                    except Exception as e:
                        logger.error(f"Request failed: {str(e)}")
                        continue

            print(f"[{opt_idx+1}/{len(opts)}] {opt} used time: ", total_time)
            
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
