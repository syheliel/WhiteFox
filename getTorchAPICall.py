from getAST import ModuleAnalyzer
from pathlib import Path
from typing import List
from loguru import logger
from src.conf import TORCH_BASE

def get_torch_api_calls(analyzer: ModuleAnalyzer, file_path: Path) -> List[str]:
    analyzer.analyze_file(file_path)
    functions = analyzer.module_info["files"]
    torch_api_calls: List[str] = []
    
    # Process each file in the module
    for file_path_str, file_info in functions.items():
        # Process each function in the file
        for func_name, _ in file_info["functions"].items():
            # The call_graph contains function calls
            # Format is "file_path:function_name" -> set of called functions
            caller_key = f"{file_path_str}:{func_name}"
            if caller_key in analyzer.module_info["call_graph"]:
                for api_call in analyzer.module_info["call_graph"][caller_key]:
                    if "torch" in api_call:
                        torch_api_calls.append(api_call)
    
    return torch_api_calls

if __name__ == "__main__":
    inductor_base = Path(TORCH_BASE) / "torch" / "_inductor"
    analyzer = ModuleAnalyzer(TORCH_BASE, inductor_base)
    torch_api_calls = get_torch_api_calls(analyzer, inductor_base / "aoti_eager.py")
    function_name = "extract_tensor_metadata"
    print("Module call graph:")
    for module, calls in analyzer.module_info["module_call_graph"].items():
        print(f"\nModule: {module}")
        for caller, callees in calls.items():
            print(f"  {caller} calls:")
            for callee in callees:
                print(f"    - {callee}")
    print(torch_api_calls)
