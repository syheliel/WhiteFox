"""
This file runs the template_exec.py and restart it when it crashes.
"""

from torch_exec.template_exec import ResType
from copy import deepcopy
import time
from pathlib import Path
import os
import tqdm
from torch_exec.types import Seed
import src.conf  # type: ignore
import torch
# from coverage import Coverage  # Original subprocess based coverage
from coverage import Coverage # API based coverage
from loguru import logger
import click
from typing import List
from torch_exec.template_exec import execute_single_file


    
def run_process(cov: Coverage, seed: Seed, result_dir: Path, device: str, timeout: int) -> None: # Timeout is currently unused
    """Run the target logic as a direct function call"""
    cur_test_target = seed["seed_path"].name
    logger.info(f"Executing function for: {cur_test_target}")
    try:
        result, error_msg = execute_single_file( # type: ignore
            target_file_path=seed["seed_path"],
            device=device,
            temp_dir=result_dir,
            api_name=cur_test_target
        )
        logger.info(f"Function returned result: {result.name}")
        logger.info(f"Function returned error: {error_msg}")
        
    except Exception as e:
        logger.error(f"Exception during function call for {cur_test_target}: {e}")
        raise e


@click.command()
@click.option(
    "--input-dir", type=str, required=True, help="the input dir of generated test cases"
)
@click.option(
    "--result-dir", "--output-dir", "--out-dir", type=str, default="result-4-default", help="the result dir to store outputs"
)
@click.option("--device", type=str, default="cuda", help="the backend device to test")
@click.option("--timeout", type=int, default=20, help="timeout in seconds")
@click.option("--fuzz-time", type=int, default=3600, help="fuzz time in seconds")
@click.option("--num-threads", type=int, default=8, help="number of threads to use for parallel processing")
def main(
    input_dir: str,
    result_dir: str,
    device: str,
    timeout: int,
    fuzz_time: int,
    num_threads: int,
) -> None:
    # Setup directories
    input_p:Path = Path(input_dir)
    new_seeds:List[Seed] = [{"seed_path": p} for p in list(input_p.glob("**/*.py"))]
    old_seeds:List[Seed] = deepcopy(new_seeds)
    res_name = f"{input_p.name}-{device}"
    result_p = Path(result_dir) / res_name
    result_p.mkdir(parents=True, exist_ok=True)
    logger.info(f"Found {len(new_seeds)} seeds")
    
    # loop until fuzz_time
    start_time = time.time()
    cov = Coverage(source=["torch"])
    cov.start()
    while time.time() - start_time < fuzz_time:
        new_seed = new_seeds.pop(0)
        # run the seed
        t = torch.tensor([0])
        torch.abs(t)
        run_process(cov, new_seed, result_p, device, timeout)
        old_seeds.append(new_seed)
        print(cov.report())
    
if __name__ == "__main__":
    main()
