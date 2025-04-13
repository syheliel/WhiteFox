import click
import tempfile
from torch_exec.ProcessStatus import ProcessStatus
import src.conf  # type: ignore
import astunparse  # type: ignore
import os
from pathlib import Path
from typing import List, NamedTuple, TypedDict, Union, Any
import ast
import traceback
import torch
from torch_exec.torch_utils import test_wrapper
from torch_exec.constant.returntypes import ResType
from loguru import logger



# Define a TypedDict to replace global variables
class Config(TypedDict):
    api_dir: Path
    result_dir: Path
    gencode_dir: Path
    temp_dir: Path # template directory to store code
    device: str
    cov: bool
    output_limit: int
    seed: int
    match_cov_file: Path
    validate: bool


def clean_match_cov(config: Config) -> None:
    config["match_cov_file"].write_text("")


def get_match_cov(config: Config) -> List[str]:
    cov = []
    if config["match_cov_file"].exists():
        cov = config["match_cov_file"].read_text().splitlines()
    clean_match_cov(config)
    return cov


class MultilineAssignTransformer(ast.NodeTransformer):
    def visit_Assign(self, node: ast.Assign) -> Union[ast.Assign, List[ast.Assign]]:
        if isinstance(node.targets[0], ast.Tuple) and isinstance(node.value, ast.Tuple):
            if len(node.targets[0].elts) == len(node.value.elts):
                return [
                    ast.Assign(targets=[t], value=v)
                    for t, v in zip(node.targets[0].elts, node.value.elts)
                ]
        return node


class LibAssignRemover(ast.NodeTransformer):
    def __init__(self) -> None:
        self.lib_name = "torch"
        super().__init__()

    def visit_Assign(self, node: ast.Assign) -> Union[ast.Assign, ast.Pass]:
        if any(self.is_lib_attribute(target) for target in node.targets):
            return ast.Pass()
        return self.generic_visit(node) # type: ignore

    def is_lib_attribute(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == self.lib_name:
                return True
            return self.is_lib_attribute(node.value)
        return False


class CodeParser:
    transformers: List[ast.NodeTransformer]
    lib_name: str
    imports: str
    _init_code: str
    config: Config
    @staticmethod
    def is_input(x: Any) -> bool:
        return torch.is_tensor(x)

    def __init__(self, config: Config) -> None:
        self.transformers = [MultilineAssignTransformer(), LibAssignRemover()]
        self.imports = (
            "import os\nimport torch\nimport torch.nn.functional as F\nimport torch.nn as nn\n"
            "import numpy as np\nfrom torch.autograd import Variable\nimport math\n"
            "import torch as th\nimport torch.linalg as la\n"
            "from torch.nn import Parameter\n"
            "import torch.linalg as linalg\n"
        )
        self._init_code = "{} = torch.randn(1, 1, 1)\n"
        self.config = config

    def input_init_code(self, arg_name: str) -> str:
        return self._init_code.format(arg_name)

    def split_func_tensor(self, code: str) -> tuple[str, List[str], str]:
        # get the code of model
        code = self.preprocessing(code)
        # logger.info(f"code: {code}")
        tree = ast.parse(code)

        class_init_args: List[str] = []
        class_init_required_args: List[str] = []
        class_init_code: str = ""

        class_code: str = ""
        class_name: str = ""

        class_forward_args: List[str] = []
        class_forward_required_args: List[str] = []

        inputs: List[str] = []
        input_init_code: str = ""

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
                except Exception:
                    logger.error(f"can't find __init__ method in {class_name}")
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
                except Exception:
                    logger.error(f"can't find forward method in {class_name}")
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
                                "func = "
                                + ast.unparse(value)
                                + f".to('{self.config['device']}')\n"
                            )
                        else:
                            class_init_code = ""
                        continue

                    try:
                        tgt = node.targets[0].id  # type: ignore
                    except Exception:
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
                            if self.is_input(eval(tgt)): # type: ignore
                                inputs.append(tgt) # type: ignore
                                input_init_code += init_code + "\n"
                            elif tgt in class_forward_args:
                                inputs.append(tgt) # type: ignore
                                input_init_code += init_code + "\n"
                        except Exception:
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
            class_init_code += f"\nfunc = {class_name}({', '.join(class_init_required_args)}).to('{self.config['device']}')\n"
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

    def preprocessing(self, _code: str) -> str:
        code = _code.replace("\t", "    ")

        new_lines: List[str] = []
        for line in code.splitlines():
            if line.strip().startswith("assert"):
                continue
            new_lines.append(line)
        code = "\n".join(new_lines)

        tree = ast.parse(code)
        for transformer in self.transformers:
            tree = transformer.visit(tree)
        code: str = astunparse.unparse(tree)  # type: ignore
        # logger.info(f"code: {code}")
        code = code.replace("[(","[").replace(")]","]")
        code = code.replace("(:", ":").replace(":)", ":")
        return code

    @staticmethod
    def find_name_in_tree(
        tree: ast.AST, arg_name: str, use_default: bool = False
    ) -> str:
        for _n in tree.body: # type: ignore
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



def _cross_check(
    func_def_code: str, tensors: List[str], api_name: str, config: Config
) -> None:
    """use exception to handle the result, see core_loop"""
    logger.info(f"cross checking {api_name}")
    func_def_code += f"test_inputs = [{', '.join(tensors)}]\n"

    if config["cov"]:
        clean_match_cov(config)

    gencode_file = config["gencode_dir"] / (api_name + ".py")
    gencode_file.parent.mkdir(parents=True, exist_ok=True)
    gencode_file.touch(exist_ok=True)
    result, errors = test_wrapper(
        func_def_code, config["seed"], tensors, config["device"], gencode_file)


    error_msg = "\n".join([f"{k}:\n{v}\n" for k, v in errors.items()])
    error_msg = "\n'''\n" + error_msg + "'''"

    # print(msg)
    if result == ResType.PASS:
        with open(config["result_dir"] / "success.log", "a") as fw:
            fw.write(api_name + "\n")
        raise Exception("Success", "succeed")
    elif result == ResType.NAN:
        with open(config["result_dir"] / "nan.log", "a") as fw:
            fw.write(api_name + "\n")
        raise Exception("NAN", "void")
    elif result == ResType.RANDOM:
        with open(config["result_dir"] / "random.log", "a") as fw:
            fw.write(api_name + "\n")
        raise Exception("RANDOM", "void")
    elif result == ResType.SKIP:
        skip_dir = config["result_dir"] / "skip"

        os.makedirs(skip_dir, exist_ok=True)
        with open(f"{skip_dir}/{api_name}", "w") as fw:
            fw.write(func_def_code)
        with open(config["result_dir"] / "skip.log", "a") as fw:
            fw.write(api_name + "\n")

        raise Exception("SKIP", "void")
    else:
        exception_name = str(result).replace("ResType.", "")
        bug_dir = config["result_dir"] / exception_name.lower()

        os.makedirs(bug_dir, exist_ok=True)
        with open(config["result_dir"] / f"{exception_name.lower()}.log", "a") as fw:
            fw.write(api_name + "\n")

        with open(f"{bug_dir}/{api_name}", "w") as fw:
            fw.write(func_def_code + "\n# " + exception_name + error_msg)
        raise Exception(exception_name, "Catch")


def validate(
    func_def_code: str, tensors: List[str], api_name: str, config: Config
) -> None:
    logger.info(f"validating {api_name}")
    func_def_code += f"test_inputs = [{', '.join(tensors)}]\n"

    result, _ = test_wrapper(
        func_def_code, config["seed"], tensors, config["device"], config["gencode_dir"] / (api_name + ".py"), "validate"
    )

    if result == ResType.PASS:
        with open(config["result_dir"] / "success.log", "a") as fw:
            fw.write(api_name + "\n")
    else:
        with open(config["result_dir"] / "fail.log", "a") as fw:
            fw.write(api_name + "\n")


class Task(NamedTuple):
    opt: str
    label: str
    code: str


def core_oracle(
    config: Config, code: str, api_name: str, is_validate: bool = False
) -> None:
    logger.info(f"testing {api_name}")
    codeParser = CodeParser(config)
    example_p = config["temp_dir"] / "origin_code.py"
    example_p.touch(exist_ok=True)
    example_p.write_text("# origin code:\n" + code + '\n')
    logger.info(f"example_p: {example_p}")
    class_def_code, inputs, input_init_code = codeParser.split_func_tensor(code)
    imports = codeParser.imports

    if len(inputs) == 0:
        inputs.append("input_tensor")
        input_init_code += codeParser.input_init_code("input_tensor")

    class_def_code = imports + "\n" + class_def_code + "\n" + input_init_code + "\n"

    if is_validate:
        validate(class_def_code, inputs, api_name, config)
    else:
        _cross_check(class_def_code, inputs, api_name, config)


def core_loop(config: Config) -> None:
    target_files: List[Path] = []
    for target_file in config["api_dir"].iterdir():
        if not target_file.name.endswith(".py"):
            continue
        target_files.append(target_file)
    logger.info(f"found {len(target_files)} target files")
    for target_file in target_files:
        code = target_file.read_text()
        try:
            core_oracle(config, code, target_file.name, is_validate=config["validate"])
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
                logger.info("exit with code {}".format(ProcessStatus.RETRY))
                exit(-1) 

            if "Catch" in reason:
                with open(config["result_dir"] / "catches.log", "a") as f:
                    f.write("\nLmfuzzTestcase target:{} reason:{} seed:{} detail:{}".format(
                            target_file.name, reason, config["seed"], detail
                        )
                    )
            logger.info(
                "LmfuzzTestcase target:{} reason:{} seed:{} detail:{}".format(
                    target_file.name, reason, config["seed"], detail
                )
            )



@click.command()
@click.option("--api-dir", type=str, required=True, help="the dir to store code genereate for one api")
@click.option("--res-dir", type=str, required=True, help="the dir to store result")
@click.option("--cov", is_flag=True, default=False, help="Enable coverage tracking")
@click.option("--validate", is_flag=True, default=False, help="Run in validation mode")
@click.option(
    "--device", type=str, default="cpu", help="Device to run on (cpu, cuda, etc.)"
)
def main(
    api_dir: str, res_dir: str, cov: bool, validate: bool, device: str
) -> None:
    """Template execution script for testing PyTorch models."""
    api_p = Path(api_dir)
    if not api_p.exists():
        raise FileNotFoundError(f"API directory {api_dir} does not exist")
    res_p = Path(res_dir)
    res_p.mkdir(parents=True, exist_ok=True)
    tempdir = tempfile.mkdtemp()
    gencode_p:Path = res_p / "gencode"
    config: Config = {
        "api_dir": api_p,
        "result_dir": res_p,
        "temp_dir": Path(tempdir),
        "device": device,
        "cov": cov,
        "validate": validate,
        "seed": 420,
        "output_limit": 1024,
        "match_cov_file": Path("/tmp/match_trigger.log"),
        "gencode_dir": gencode_p,
    }


    if config["cov"]:
        config["match_cov_file"] = Path(
            "/", "tmp", f"trigger-{config['result_dir'].name}"
        )
        torch.version.log_path = str(config["match_cov_file"])  # type: ignore

    # Run the core loop with the config
    core_loop(config)
    # Some sneaky code may contain exit(0) or other equivalent calls
    # We distinguish ourselves from them with a magic number
    logger.info("exit with code {}".format(ProcessStatus.FINISH))
    exit(ProcessStatus.FINISH)


if __name__ == "__main__":
    main()

