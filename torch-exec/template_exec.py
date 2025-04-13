import click
import src.conf
import astunparse
import json
import os
from pathlib import Path
from typing import List, Optional, Any, Callable, NamedTuple
import ast
import traceback
import torch
from torch_utils import test_wrapper
from loguru import logger

from constant.returntypes import ResType


# default values to make lint check happy :)
OUT_DIR: Path = Path("out-4")
RESULT_DIR: Path = Path("result-4")
TEST_DIR: Path = Path("test-4")
TEST_LOG_PATH: Path = TEST_DIR / "tested.log"
TEMP_LOG_PATH: Path = TEST_DIR / "atemp.py"
DEVICE: str = "cpu"
COV: bool = False

OUTPUT_LIMIT: int = 1024
SEED: int = 420
MATCH_COV_FILE: Path = Path("/tmp/match_trigger.log")
MAXIMUM_TESTCASES = 10


def clean_match_cov():
    MATCH_COV_FILE.write_text("")


def get_match_cov():
    cov = []
    if MATCH_COV_FILE.exists():
        cov = MATCH_COV_FILE.read_text().splitlines()
    clean_match_cov()
    return cov


class MultilineAssignTransformer(ast.NodeTransformer):
    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Tuple) and isinstance(node.value, ast.Tuple):
            if len(node.targets[0].elts) == len(node.value.elts):
                return [
                    ast.Assign(targets=[t], value=v)
                    for t, v in zip(node.targets[0].elts, node.value.elts)
                ]
        return node


class LibAssignRemover(ast.NodeTransformer):
    def __init__(self, lib_name: str = "torch") -> None:
        super().__init__()
        self.lib_name = lib_name

    def visit_Assign(self, node):
        if any(self.is_lib_attribute(target) for target in node.targets):
            return ast.Pass()
        return self.generic_visit(node)

    def is_lib_attribute(self, node):
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == self.lib_name:
                return True
            return self.is_lib_attribute(node.value)
        return False


class CodeParser:
    def __init__(self, lib_name: str = "torch") -> None:
        self.transformers = [MultilineAssignTransformer(), LibAssignRemover(lib_name)]
        self.lib_name = lib_name
        if lib_name == "torch":
            self.is_input = lambda x: torch.is_tensor(x)
            self.imports = (
                "import os\nimport torch\nimport torch.nn.functional as F\nimport torch.nn as nn\n"
                "import numpy as np\nfrom torch.autograd import Variable\nimport math\n"
                "import torch as th\nimport torch.linalg as la\n"
                "from torch.nn import Parameter\n"
                "import torch.linalg as linalg\n"
            )
            self._init_code = "{} = torch.randn(1, 1, 1)\n"
        elif lib_name == "tf":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def input_init_code(self, arg_name):
        return self._init_code.format(arg_name)

    def split_func_tensor(self, code):
        # get the code of model
        code = self.preprocessing(code)
        tree = ast.parse(code)

        class_init_args = []
        class_init_required_args = []
        class_init_code = ""

        class_code = ""
        class_name = ""

        class_forward_args = []
        class_forward_required_args = []

        inputs: List[str] = []
        input_init_code = ""

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_code += ast.unparse(node) + "\n\n"
                class_name = node.name

                # get the arguments the initiation of this class
                try:
                    init_method = next(
                        node
                        for node in ast.walk(node)
                        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
                    )

                    class_init_args = [arg.arg for arg in init_method.args.args[1:]]
                    defaults = init_method.args.defaults
                    class_init_required_args = class_init_args[
                        : len(class_init_args) - len(defaults)
                    ]
                except Exception as e:
                    pass

                try:
                    forward_method = next(
                        node
                        for node in ast.walk(node)
                        if isinstance(node, ast.FunctionDef) and node.name == "forward"
                    )
                    class_forward_args = [
                        arg.arg for arg in forward_method.args.args[1:]
                    ]
                    defaults = forward_method.args.defaults
                    class_forward_required_args = class_forward_args[
                        : len(class_forward_args) - len(defaults)
                    ]
                except Exception as e:
                    pass

            elif isinstance(node, ast.Assign):
                value = node.value
                if isinstance(value, ast.Call):
                    # first check whether is initialization of the class
                    if isinstance(value.func, ast.Name) and value.func.id == class_name:
                        # first split the tensor arguments and non-tensor arguments
                        if len(value.args) >= len(class_init_required_args) and len(
                            value.args
                        ) <= len(class_init_args):
                            class_init_code = (
                                "func = " + ast.unparse(value) + f".to('{DEVICE}')\n"
                            )
                        else:
                            class_init_code = ""
                        continue

                    func = value.func
                    args = value.args

                    try:
                        tgt = node.targets[0].id # type: ignore
                    except Exception as e:
                        continue

                    init_code = ast.unparse(node)
                    if tgt not in inputs:
                        # we need the arg code
                        for arg in ast.walk(value):
                            if isinstance(arg, ast.Name):
                                init_code = (
                                    self.find_name_in_tree(tree, arg.id)
                                    + "\n"
                                    + init_code
                                )
                            elif isinstance(arg, ast.Starred):
                                if isinstance(arg.value, ast.Name):
                                    init_code = (
                                        self.find_name_in_tree(tree, arg.value.id)
                                        + "\n"
                                        + init_code
                                    )

                        # test whether is tensor
                        try:
                            exec(init_code)
                            if self.is_input(eval(tgt)):
                                inputs.append(tgt)
                                input_init_code += init_code + "\n"
                            elif tgt in class_forward_args:
                                inputs.append(tgt)
                                input_init_code += init_code + "\n"
                        except Exception as e:
                            pass

        class_init_args_code = ""
        for arg_name in class_init_required_args:
            class_init_args_code += (
                self.find_name_in_tree(tree, arg_name, use_default=True) + "\n"
            )
        if class_init_code != "":
            class_init_code = class_init_args_code + class_init_code
        else:
            class_init_code = class_init_args_code
            class_init_code += f"\nfunc = {class_name}({', '.join(class_init_required_args)}).to('{DEVICE}')\n"
        class_code += "\n" + class_init_code

        if len(inputs) < len(class_forward_args):
            diff = len(class_forward_args) - len(inputs)
            for arg_name in class_forward_required_args:
                if arg_name not in inputs:
                    inputs.append(arg_name)
                    input_init_code += f"{arg_name} = 1\n"
                    diff -= 1
                    if diff == 0:
                        break

        return class_code, inputs, input_init_code

    def preprocessing(self, code: str):
        code = code.replace("\t", "    ")

        new_lines = []
        for line in code.splitlines():
            if line.strip().startswith("assert"):
                continue
            new_lines.append(line)
        code = "\n".join(new_lines)

        tree = ast.parse(code)
        for transformer in self.transformers:
            tree = transformer.visit(tree)
        code = astunparse.unparse(tree)
        code = code.replace("(:", ":").replace(":)", ":")
        return code

    @staticmethod
    def find_name_in_tree(tree, arg_name, use_default=False):
        for _n in tree.body:
            if isinstance(_n, ast.Assign):
                for _t in _n.targets:
                    if isinstance(_t, ast.Name) and _t.id == arg_name:
                        return ast.unparse(_n)
        if arg_name == "batch_size":
            return f"{arg_name} = 1"

        if use_default:
            return f"{arg_name} = 1"
        else:
            return ""

CODE_PARSER: CodeParser = CodeParser("torch")

def _cross_check(func_def_code, tensors, filename):
    logger.info(f"cross checking {filename}")
    func_def_code += f"test_inputs = [{', '.join(tensors)}]\n"
    TEMP_LOG_PATH.write_text(func_def_code)

    if COV:
        clean_match_cov()

    
    result, errors = test_wrapper(func_def_code, 420, tensors, DEVICE)

    if COV:
        match_info = {filename: get_match_cov()}
        with open(TEST_DIR / "match.log", "a") as fw:
            fw.write(json.dumps(match_info) + "\n")

    error_msg = "\n".join([f"{k}:\n{v}\n" for k, v in errors.items()])
    error_msg = "\n'''\n" + error_msg + "'''"

    # print(msg)
    if result == ResType.PASS:
        with open(TEST_DIR / "success.log", "a") as fw:
            fw.write(filename + "\n")
        raise Exception("Success", "succeed")
    elif result == ResType.NAN:
        with open(TEST_DIR / "nan.log", "a") as fw:
            fw.write(filename + "\n")
        raise Exception("NAN", "void")
    elif result == ResType.RANDOM:
        with open(TEST_DIR / "random.log", "a") as fw:
            fw.write(filename + "\n")
        raise Exception("RANDOM", "void")
    elif result == ResType.SKIP:
        skip_dir = RESULT_DIR / "skip"
        os.makedirs(skip_dir, exist_ok=True)
        with open(f"{skip_dir}/{filename}", "w") as fw:
            fw.write(func_def_code)
        with open(TEST_DIR / "skip.log", "a") as fw:
            fw.write(filename + "\n")

        raise Exception("SKIP", "void")
    else:
        exception_name = str(result).replace("ResType.", "")
        bug_dir = RESULT_DIR / exception_name.lower()

        os.makedirs(bug_dir, exist_ok=True)
        with open(TEST_DIR / f"{exception_name.lower()}.log", "a") as fw:
            fw.write(filename + "\n")

        with open(f"{bug_dir}/{filename}", "w") as fw:
            fw.write(func_def_code + "\n# " + exception_name + error_msg)
        raise Exception(exception_name, "Catch")


def validate(func_def_code:str, tensors:List[str], filename:str):
    logger.info(f"validating {filename}")
    func_def_code += f"test_inputs = [{', '.join(tensors)}]\n"
    TEMP_LOG_PATH.write_text(func_def_code)

    result, errors = test_wrapper(func_def_code, 420, tensors, DEVICE, 'validate')
    
    if result == ResType.PASS:
        with open(TEST_DIR / "success.log", "a") as fw:
            fw.write(filename + "\n")
    else:
        with open(TEST_DIR / "fail.log", "a") as fw:
            fw.write(filename + "\n")

class Task(NamedTuple):
    opt: str
    label: str
    code: str

def read_all_tasks() -> List[Task]:
    """
    Reads all task files from the output directory structure.
    
    This function scans the OUT_DIR directory for subdirectories, each representing an optimization option.
    Within each subdirectory, it finds Python files (.py) and reads their contents.
    
    Returns:
        list: A sorted list of tasks, where each task is a list containing:
            - opt (str): The optimization option directory name
            - label (str): The filename without the .py extension
            - code (str): The contents of the Python file
            
    The returned list is sorted first by optimization option and then by label.
    """
    tasks = []
    for opt_dir in OUT_DIR.iterdir():
        if not opt_dir.is_dir():
            continue
        opt = opt_dir.name
        for filename in opt_dir.iterdir():
            if not filename.name.endswith(".py"):
                continue

            label = filename.name[:-3]
            code = filename.read_text()
            tasks.append([opt, label, code])

    tasks = sorted(tasks, key=lambda x: (x[0], x[1]))
    return [Task(opt, label, code) for opt, label, code in tasks]


def core_oracle(code:str, filename:str, is_validate:bool=False):
    logger.info(f"testing {filename}")
    class_def_code, inputs, input_init_code = CODE_PARSER.split_func_tensor(code)
    logger.info(f"the length of class_def_code: {len(class_def_code)}")
    logger.info(f"the length of inputs: {len(inputs)}")
    logger.info(f"the length of input_init_code: {len(input_init_code)}")
    imports = CODE_PARSER.imports

    if len(inputs) == 0:
        inputs.append("input_tensor")
        input_init_code += CODE_PARSER.input_init_code("input_tensor")

    class_def_code = imports + "\n" + class_def_code + "\n" + input_init_code + "\n"

    if is_validate:
        validate(class_def_code, inputs, filename)
    else:
        _cross_check(class_def_code, inputs, filename)


def core_loop(args):
    tasks:List[Task] = read_all_tasks()
    logger.info("all tasks:")
    logger.info(f"read tested from: {TEST_LOG_PATH}")
    if TEST_LOG_PATH.exists():
        tested = set(open(TEST_LOG_PATH, "r").read().splitlines())
    else:
        tested = set([])

    count = 0
    for id in range(len(tasks)):
        task = tasks[id]
        api, label, code = task
        filename = label + ".py"

        if filename in tested:
            logger.info(f"Skipping {filename} because it has already been tested")
            continue
        with open(TEST_LOG_PATH, "a") as fw:
            fw.write(filename + "\n")

        try:
            core_oracle(code, filename, is_validate=args.validate)
        except Exception as e:
            reason: str = "FrameworkCrashCatch"
            detail: str = str(e)
            if len(e.args) >= 2:
                reason: str = e.args[0]
                detail: str = e.args[1]

            if (
                reason == "FrameworkCrashCatch"
            ):  # FrameworkCrashCatch is printed by driver
                logger.error(traceback.format_exc())
                logger.error(detail)
                exit(-1)

            if "Catch" in reason:
                with open("catches.log", "a") as f:
                    f.write(
                        "\nLmfuzzTestcase {} {} {} {} {} {}".format(
                            id, api, label, reason, SEED, detail
                        )
                    )
            logger.info("\nLmfuzzTestcase id:{} api:{} label:{} reason:{} seed:{} detail:{}".format(id, api, label, reason, SEED, detail))
            logger.info("----------------------------------\n")

        count += 1
        if count >= MAXIMUM_TESTCASES:
            exit(123)


@click.command()
@click.option('--out-dir', type=str, default="out-5", help='Output directory')
@click.option('--res-dir', type=str, default="res-5", help='Result directory')
@click.option('--test-dir', type=str, default="test-5", help='Test directory')
@click.option('--cov', is_flag=True, default=False, help='Enable coverage tracking')
@click.option('--validate', is_flag=True, default=False, help='Run in validation mode')
@click.option('--device', type=str, default='cpu', help='Device to run on (cpu, cuda, etc.)')
def main(out_dir, res_dir, test_dir , cov, validate, device):
    """Template execution script for testing PyTorch models."""
    global OUT_DIR, RESULT_DIR, TEST_DIR, TEST_LOG_PATH, TEMP_LOG_PATH, DEVICE, COV, MATCH_COV_FILE
    OUT_DIR = Path(out_dir)
    RESULT_DIR = Path(res_dir)
    TEST_DIR = Path(test_dir)
    TEST_LOG_PATH = TEST_DIR / "tested.log"
    TEMP_LOG_PATH = TEST_DIR / "atemp.py"
    CRASH_LOG_PATH = TEST_DIR / "crash.log"
    DEVICE = device
    COV = cov
    logger.info(f"OUT_DIR: {OUT_DIR}")
    logger.info(f"RESULT_DIR: {RESULT_DIR}")
    logger.info(f"TEST_DIR: {TEST_DIR}")
    logger.info(f"TEST_LOG_PATH: {TEST_LOG_PATH}")
    logger.info(f"TEMP_LOG_PATH: {TEMP_LOG_PATH}")
    logger.info(f"CRASH_LOG_PATH: {CRASH_LOG_PATH}")

    # Ensure directories exist
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    CODE_PARSER = CodeParser(lib_name="torch")
    if COV:
        MATCH_COV_FILE = Path("/", "tmp", f"trigger-{RESULT_DIR.name}")
        torch.version.log_path = str(MATCH_COV_FILE) # type: ignore

    # Create a simple args object to maintain compatibility with core_loop
    class Args:
        def __init__(self):
            self.validate = False
    
    args = Args()
    args.validate = validate
    
    core_loop(args)


if __name__ == "__main__":
    main()
    # Some sneaky code may contain exit(0) or other equivalent calls
    # We distinguish ourselves from them with a magic number
    exit(233)
