from getAST import extract_function_apis
from pathlib import Path
from typing import List
from loguru import logger
from conf import TORCH_BASE
def get_torch_api_calls(file_path: Path) -> List[str]:
    functions = extract_function_apis(file_path)
    torch_api_calls = []
    for func_name, func_info in functions.items():
        for api_call in func_info["api_calls"]:
            # if "torch" in api_call:
            torch_api_calls.append(api_call)
    return torch_api_calls

if __name__ == "__main__":
    inductor_base = Path(TORCH_BASE) / "torch" / "_inductor"
    for file in inductor_base.glob("**/*.py"):
        logger.info(f"Processing {file}")
        torch_api_calls = get_torch_api_calls(file)
        logger.info(f"torch_api_calls: {torch_api_calls}")
        
