"""
This file runs the template_exec.py and restart it when it crashes.
"""
import subprocess as sp
import time
from pathlib import Path
import os
import torch
from loguru import logger
import click

TEST_DIR = Path("test_dir")
RESULT_DIR = Path("result_dir")


def get_last_tested():
    tested_path = TEST_DIR / "tested.log"
    if tested_path.exists() == False:
        return "start"

    text = tested_path.read_text()
    lines = text.splitlines()
    if len(lines) < 2:
        return "start"
    else:
        return lines[-2]


def combine_cov(cov_dir, cov_datafile):
    combine_cmds = [
        "coverage",
        "combine",
        f"--data-file={cov_datafile}",
        os.path.join(cov_dir, ".coverage.*"),
    ]
    output = sp.run(" ".join(combine_cmds), stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
    if output.returncode != 0:
        logger.error("combine coverage failed")
        logger.error(output.stderr.decode())
        return


def collect_cov(cov_datafile):
    cov_jsonfile = cov_datafile.with_suffix(".json")
    ret = sp.run(
        [
            "python",
            "-m",
            "coverage",
            "json",
            f"--data-file={cov_datafile}",
            "-o",
            str(cov_jsonfile),
            "--pretty-print",
        ],
    )
    if ret.returncode != 0:
        logger.error("collect coverage failed")
        return


@click.command()
@click.option(
    "--input-dir",
    type=str,
    required=True,
    help="the input dir of generated test cases",
)
@click.option(
    "--res-dir",
    type=str,
    default="_results",
    help="the result dir to store outputs",
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    help="the backend device to test",
)
@click.option(
    "--timeout",
    type=int,
    default=20,
    help="timeout in seconds",
)
@click.option(
    "--cov",
    is_flag=True,
    default=False,
    help="collect coverage",
)
@click.option(
    "--cover",
    is_flag=True,
    default=False,
    help="whether the inputs trigger/cover the optimization",
)
@click.option(
    "--suffix",
    type=str,
    default="",
    help="suffix for result directory",
)
@click.option(
    "--validate",
    is_flag=True,
    default=False,
    help="validate mode",
)
def main(input_dir, res_dir, device, timeout, cov, cover, suffix, validate):
    
    TIMEOUT = timeout
    DEVICE = device

    OUT_DIR = Path(input_dir)

    res_name = f"{OUT_DIR.name}-{device}"
    if suffix != "":
        res_name += f"-{suffix}"
    if validate:
        res_name += "-validate"
    RESULT_DIR = Path(res_dir) / res_name
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR = RESULT_DIR / "test"

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    log_file = RESULT_DIR / "run.log"
    err_file = RESULT_DIR / "err.log"
    log_file.write_text("Start testing\n")
    err_file.write_text("Start testing\n")

    log_file = open(log_file, "a")
    stderr_file = open(err_file, "a")
    crash_file = open(TEST_DIR / "crash.log", "a")
    timeout_file = open(RESULT_DIR / "timeout.log", "a")
    kill_file = open(RESULT_DIR / "killed.log", "a")
    start_time = time.time()

    # for coverage collection
    cov_cnt = 0
    cov_dir = Path(RESULT_DIR, "cov-datafile", "_cov_tmp_dir")
    cov_dir.mkdir(parents=True, exist_ok=True)
    cov_datafile = Path(RESULT_DIR, "cov-datafile", "my.coverage")

    while True:
        env = {
            "TORCHINDUCTOR_PERMUTE_FUSION": "1",
            "TORCHDYNAMO_VERBOSE": "1",
        }
        if not cover:
            env["TORCHINDUCTOR_SHAPE_PADDING"] = "1"
        env = {**env, **os.environ}

        if cov:
            cov_data: Path = cov_dir / f".coverage.{cov_cnt}"
            python_cmd: list[str] = [
                "python",
                "-m",
                "coverage",
                "run",
                f"--source={torch.__path__[0]}", # type: ignore
                f"--data-file={cov_data}",
                "-a",
            ]
            cov_cnt += 1
        else:
            python_cmd = ["python"]

        script_cmd: list[str] = [
            "./torch-exec/template_exec.py",
            f"--out-dir={OUT_DIR}",
            f"--res-dir={RESULT_DIR}",
            f"--test-dir={TEST_DIR}",
            f"--device={DEVICE}",
        ]
        if validate:
            script_cmd.append("--validate")
        if cov:
            python_cmd.append("--cov")

        # Log the command being executed
        final_cmd:str = " ".join(python_cmd + script_cmd)
        logger.info(f"Executing command: {final_cmd}")

        process = sp.Popen(
            python_cmd + script_cmd,
            stdout=log_file,
            stderr=stderr_file,
            env=env,
        )

        count = 0
        while True:
            cur_test_target = get_last_tested()
            logger.info(f"Current test target: {cur_test_target}")
            result = process.poll()
            if result is None:
                time.sleep(1)
                if cur_test_target != get_last_tested():
                    cur_test_target = get_last_tested()
                    count = 0
                    continue
                elif count >= TIMEOUT:
                    logger.warning("TIMEOUT, Kill process")
                    process.kill()
                    timeout_file.write(f"{cur_test_target} TIMEOUT\n")
                    timeout_file.write(str(TEST_DIR / "atemp.py") + "\n")
                    timeout_file.flush()
                else:
                    count += 1
                    continue
            elif result == 233:
                logger.info("FINISH")
                used_time = time.time() - start_time
                logger.info(f"Used time: {used_time}")

                if cov:
                    # combine the coverage
                    combine_cov(cov_dir, cov_datafile)
                    collect_cov(cov_datafile)

                # output_cov()
                log_file.write(f"\nUsed time: {used_time}")
                exit(233)
            elif result == 123:
                logger.info("Retrying ...")
            elif result == -9 or result == 255:
                # This is SIGKILL
                # We don't need to do anything
                kill_file.write(f"{cur_test_target} KILLED\n")
                kill_file.write(str(TEST_DIR / "atemp.py") + "\n")
                kill_file.flush()
                logger.warning(f"KILLED: {cur_test_target}")
            else:
                logger.error(f"Process returned code: {result}")
                crash_file.write(
                    f"\n{cur_test_target} CRASH with return code {result}\n"
                )
                try:
                    crash_file.write(str(TEST_DIR / "atemp.py") + "\n")
                except FileNotFoundError:
                    crash_file.write("No temporary log file found\n")
                crash_file.flush()
                logger.error(f"ERROR: {cur_test_target}")
            break
        used_time = time.time() - start_time
        logger.info(f"Restart at time: {used_time}")


if __name__ == "__main__":
    main()