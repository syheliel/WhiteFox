"""
This file runs the template_exec.py and restart it when it crashes.
"""

import subprocess as sp
import time
from pathlib import Path
import os
import tqdm
import src.conf  # type: ignore
import torch
from loguru import logger
import click
from typing import List, Dict, Optional, Any, Tuple
import concurrent.futures
import threading

from torch_exec.ProcessStatus import ProcessStatus

# Thread-local storage for file managers
thread_local = threading.local()

class FileManager:
    """Manages file operations for logging and tracking test progress"""
    
    def __init__(self, result_dir: Path,):
        self.result_dir = result_dir
        self.log_file = self._open_file(result_dir / "run.log", "a")
        self.err_file = self._open_file(result_dir / "err.log", "a")
        self.crash_file = self._open_file(result_dir / "crash.log", "a")
        self.timeout_file = self._open_file(result_dir / "timeout.log", "a")
        self.kill_file = self._open_file(result_dir / "killed.log", "a")
        
        # Initialize log files
        self.log_file.write("Start testing\n")
        self.err_file.write("Start testing\n")
    
    def _open_file(self, path: Path, mode: str):
        """Open a file and return the file object"""
        return open(path, mode)
    
    def get_last_tested(self) -> str:
        """Get the last tested target from the tested.log file"""
        tested_path = self.result_dir / "tested.log"
        if not tested_path.exists():
            return "start"

        text = tested_path.read_text()
        lines = text.splitlines()
        if len(lines) < 2:
            return "start"
        else:
            return lines[-2]
    
    def log_timeout(self, cur_test_target: str):
        """Log timeout information"""
        self.timeout_file.write(f"{cur_test_target} TIMEOUT\n")
        self.timeout_file.write(str(self.result_dir / "atemp.py") + "\n")
        self.timeout_file.flush()
    
    def log_killed(self, cur_test_target: str):
        """Log killed process information"""
        self.kill_file.write(f"{cur_test_target} KILLED\n")
        self.kill_file.write(str(self.result_dir / "atemp.py") + "\n")
        self.kill_file.flush()
    
    def log_crash(self, cur_test_target: str, result: int):
        """Log crash information"""
        self.crash_file.write(f"\n{cur_test_target} CRASH with return code {result}\n")
        try:
            self.crash_file.write(str(self.result_dir / "atemp.py") + "\n")
        except FileNotFoundError:
            self.crash_file.write("No temporary log file found\n")
        self.crash_file.flush()
    
    def log_time(self, used_time: float):
        """Log the time used"""
        self.log_file.write(f"\nUsed time: {used_time}")
    
    def log_stdout_stderr(self, stdout: bytes, stderr: bytes):
        """Log stdout and stderr at timeout"""
        self.log_file.write("\n=== STDOUT at TIMEOUT ===\n")
        self.log_file.write(stdout.decode())
        self.err_file.write("\n=== STDERR at TIMEOUT ===\n")
        self.err_file.write(stderr.decode())
    
    def close(self):
        """Close all file handles"""
        self.log_file.close()
        self.err_file.close()
        self.crash_file.close()
        self.timeout_file.close()
        self.kill_file.close()


class CoverageManager:
    """Manages coverage collection and reporting"""
    
    def __init__(self, result_dir: Path):
        self.result_dir = result_dir
        self.cov_dir = Path(result_dir, "cov-datafile", "_cov_tmp_dir")
        self.cov_dir.mkdir(parents=True, exist_ok=True)
        self.cov_datafile = Path(result_dir, "cov-datafile", "my.coverage")
        self.cov_cnt = 0
    
    def get_coverage_command(self) -> List[str]:
        """Get the command for running with coverage"""
        cov_data = self.cov_dir / f".coverage.{self.cov_cnt}"
        self.cov_cnt += 1
        return [
            "python",
            "-m",
            "coverage",
            "run",
            f"--source={torch.__path__[0]}",  # type: ignore
            f"--data-file={cov_data}",
            "-a",
            "-m"
        ]
    
    def combine_coverage(self):
        """Combine coverage data files"""
        combine_cmds = [
            "coverage",
            "combine",
            f"--data-file={self.cov_datafile}",
            os.path.join(self.cov_dir, ".coverage.*"),
        ]
        output = sp.run(" ".join(combine_cmds), stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
        if output.returncode != 0:
            logger.error("combine coverage failed")
            logger.error(output.stderr.decode())
            return
    
    def collect_coverage(self):
        """Collect coverage data and convert to JSON"""
        cov_jsonfile = self.cov_datafile.with_suffix(".json")
        ret = sp.run(
            [
                "poetry",
                "run",
                "python",
                "-m",
                "coverage",
                "json",
                f"--data-file={self.cov_datafile}",
                "-o",
                str(cov_jsonfile),
                "--pretty-print",
            ],
        )
        if ret.returncode != 0:
            logger.error("collect coverage failed")
            return


class ProcessManager:
    """Manages process execution and monitoring"""
    
    def __init__(self, file_manager: FileManager, coverage_manager: Optional[CoverageManager] = None):
        self.file_manager = file_manager
        self.coverage_manager = coverage_manager
        self.start_time = time.time()
        self.lock = threading.Lock()  # Add a lock for thread safety
    
    def get_environment(self, cover: bool) -> Dict[str, str]:
        """Get the environment variables for the process"""
        env: Dict[str, str] = {
            "TORCHINDUCTOR_PERMUTE_FUSION": "1",
            "TORCHDYNAMO_VERBOSE": "1",
        }
        if not cover:
            env["TORCHINDUCTOR_SHAPE_PADDING"] = "1"
        return {**env, **os.environ}
    
    def get_command(self, out_dir: Path, result_dir: Path, device: str, 
                   validate: bool, cov: bool) -> Tuple[List[str], List[str]]:
        """Get the command to execute"""
        if cov and self.coverage_manager:
            python_cmd = self.coverage_manager.get_coverage_command()
        else:
            python_cmd = ["poetry", "run","python", "-m"]
        
        script_cmd = [
            "torch_exec.template_exec",
            f"--api-dir={out_dir}",
            f"--res-dir={result_dir}",
            f"--device={device}",
        ]
        
        if validate:
            script_cmd.append("--validate")
            
        return python_cmd, script_cmd
    
    def handle_timeout(self, process: sp.Popen[Any], cur_test_target: str):
        """Handle process timeout"""
        logger.warning("TIMEOUT, Kill process")
        try:
            stdout, stderr = process.communicate(timeout=1)
            logger.error("=== STDOUT at TIMEOUT ===")
            logger.error(stdout.decode())
            logger.error("=== STDERR at TIMEOUT ===")
            logger.error(stderr.decode())
            self.file_manager.log_stdout_stderr(stdout, stderr)
        except sp.TimeoutExpired:
            pass
        process.kill()
        self.file_manager.log_timeout(cur_test_target)
    
    def handle_process_result(self, exit_code: int, cur_test_target: str):
        """Handle process result and return whether to continue the loop"""
        print(exit_code)
        with self.lock:  # Use lock to ensure thread safety
            if exit_code == ProcessStatus.FINISH.value:
                logger.info("FINISH")
                used_time = time.time() - self.start_time
                logger.info(f"Used time: {used_time}")
                
                if self.coverage_manager:
                    self.coverage_manager.combine_coverage()
                    self.coverage_manager.collect_coverage()
                    
                self.file_manager.log_time(used_time)
                
            elif exit_code == ProcessStatus.RETRY.value:
                logger.info("Retrying ...")
                
            elif exit_code in [ProcessStatus.KILLED.value, ProcessStatus.KILLED_ALT.value]:
                self.file_manager.log_killed(cur_test_target)
                logger.warning(f"KILLED: {cur_test_target}")
                
            else:
                logger.error(f"Process returned code: {exit_code}")
                self.file_manager.log_crash(cur_test_target, exit_code)
                logger.error(f"ERROR: {cur_test_target}")
    
    def run_process(self, api_dir: Path, result_dir: Path, device: str, 
                   timeout: int, validate: bool, cov: bool, cover: bool) -> None:
        """Run the process and monitor it"""
        env = self.get_environment(cover)
        python_cmd, script_cmd = self.get_command(api_dir, result_dir, device, validate, cov)
        
        # Log the command being executed
        final_cmd = " ".join(python_cmd + script_cmd)
        logger.info(f"Executing command: {final_cmd}")
        
        process = sp.Popen(
            python_cmd + script_cmd,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            env=env,
        )
        logger.info(f"Process started with pid: {process.pid}")
        
        process.wait(timeout=timeout)
        if process.returncode is None:
            self.handle_timeout(process, api_dir.name)
            process.kill()
        self.handle_process_result(process.returncode, api_dir.name)
    

def process_api_dir(api_dir: Path, result_p: Path, device: str, timeout: int, 
                   validate: bool, cov: bool, cover: bool, process_manager: ProcessManager) -> None:
    """Process a single API directory in a thread"""
    gencode_dir = result_p / api_dir.stem
    if gencode_dir.exists():
        logger.info(f"skip {api_dir.name} because it has already been tested")
        return
    
    process_manager.run_process(
        api_dir=api_dir,
        result_dir=result_p,
        device=device,
        timeout=timeout,
        validate=validate,
        cov=cov,
        cover=cover
    )

@click.command()
@click.option(
    "--input-dir", type=str, required=True, help="the input dir of generated test cases"
)
@click.option(
    "--result-dir", type=str, default="result-4-default", help="the result dir to store outputs"
)
@click.option("--device", type=str, default="cuda", help="the backend device to test")
@click.option("--timeout", type=int, default=20, help="timeout in seconds")
@click.option("--cov", is_flag=True, default=False, help="collect coverage")
@click.option(
    "--cover",
    is_flag=True,
    default=False,
    help="whether the inputs trigger/cover the optimization",
)
@click.option("--validate", is_flag=True, default=False, help="validate mode")
@click.option("--num-threads", type=int, default=8, help="number of threads to use for parallel processing")
def main(
    input_dir: str,
    result_dir: str,
    device: str,
    timeout: int,
    cov: bool,
    cover: bool,
    validate: bool,
    num_threads: int,
) -> None:
    # Setup directories
    input_p:Path = Path(input_dir)
    res_name = f"{input_p.name}-{device}"
    if validate:
        res_name += "-validate"
    result_p = Path(result_dir) / res_name
    result_p.mkdir(parents=True, exist_ok=True)

    api_dirs:List[Path] = list(input_p.iterdir())
    logger.info(f"Found {len(api_dirs)} API directories")
    
    # Initialize managers
    file_manager = FileManager(result_p)
    coverage_manager = CoverageManager(result_p) if cov else None
    process_manager = ProcessManager(file_manager, coverage_manager)
    
    try:
        # Run the process using a thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks to the executor
            futures = [
                executor.submit(
                    process_api_dir,
                    api_dir=api_dir,
                    result_p=result_p,
                    device=device,
                    timeout=timeout,
                    validate=validate,
                    cov=cov,
                    cover=cover,
                    process_manager=process_manager
                )
                for api_dir in api_dirs
            ]
            
            # Wait for all tasks to complete with a progress bar
            for _ in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                pass
    finally:
        # Clean up
        file_manager.close()


if __name__ == "__main__":
    main()
