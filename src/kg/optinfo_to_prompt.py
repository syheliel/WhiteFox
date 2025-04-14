from src.conf import OPTINFO_PATH, TORCH_BASE
import yaml
from pathlib import Path
from typing import List, Dict, Any, Union
from loguru import logger
import click
import tqdm
from src.llm_client import get_openai, ALL_MODEL
def optinfo_to_prompt(optinfo_path: Path = OPTINFO_PATH) -> Dict[str, Any]: # type: ignore
    with open(optinfo_path, "r") as f:
        return yaml.safe_load(f.read().replace("\t", "    "))

def dict_to_paths(d: Dict[str, Union[Dict[str, Any], List[str]]], base_path:Path=TORCH_BASE) -> List[Path]:
    """
    Convert a nested dictionary structure into a list of complete file paths.
    
    Args:
        d: The dictionary to process
        base_path: The base path to prepend to all paths
        
    Returns:
        A list of complete file paths
    """
    paths:List[Path] = []
    
    for key, value in d.items():
        current_path = base_path / key
        
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    # This is a directory with nested files
                    paths.extend(dict_to_paths(item, current_path))
                else:
                    # This is a file
                    paths.append(current_path / item)
        elif isinstance(value, dict): # type: ignore
            # This is a directory with nested files
            paths.extend(dict_to_paths(value, current_path))
    
    return paths

def get_complete_paths(optinfo_path: Path = OPTINFO_PATH) -> List[Path]:
    """
    Get a list of complete file paths from the optinfo file.
    
    Args:
        optinfo_path: Path to the optinfo file
        
    Returns:
        A list of complete file paths
    """
    optinfo = optinfo_to_prompt(optinfo_path)
    return dict_to_paths(optinfo)

def remove_unwanted_paths(paths: List[Path]) -> List[Path]:
    new_paths:List[Path] = []
    for path in paths:
        if "__init__" in path.name:
            continue
        new_paths.append(path)
    return new_paths

def path_to_modulename(path: Path, base_path:Path=TORCH_BASE) -> str:
    return path.relative_to(base_path).with_suffix("").as_posix().replace("/", ".")

SYSTEM_PROMPT = r"""You are an expert in pytorch bug hunting. I will give you a specific source code from pytorch. Please find all function that is 1. vuluable 2. has precision problem 3. inproper argument check 4. quantization or type inference problem. I will give you the example output as json format:
```json
{
	"BatchPointwiseOpsPreGradFusion": {
		"hints": [
			{
				"type": "trigger",
				"target_line": "module.graph.erase_node(node)",
				"func": "BatchPointwiseOpsPreGradFusion",
			}
		]
	},
	"BatchLinearLHSFusion": {
		"hints": [
			{
				"type": "trigger",
				"target_line": "module.graph.erase_node(node)",
				"func": "BatchLinearLHSFusion",
			}
		]
	},
}
```"""
@click.command()
@click.option("--model", type=click.Choice(ALL_MODEL), default="deepseek-v3-250324", help="Model to use for generate prompt")
@click.option("--output-dir", type=click.Path(file_okay=False,dir_okay=True, writable=True), default="./prompt-1-new/", help="Output file")
def main(model: str, output_dir: str): # type: ignore
    llm = get_openai(model)
    output_dir:Path = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = get_complete_paths()
    paths = remove_unwanted_paths(paths)[:100]
    logger.info(f"Found {len(paths)} paths")
    not_found:List[Path] = []
    for path in tqdm.tqdm(sorted(paths)):
        if not path.exists() or not path.is_file():
            logger.info(f"skip {path_to_modulename(path)}")
            not_found.append(path)
            continue
        if output_dir / f"{path_to_modulename(path)}.json" in output_dir.iterdir():
            logger.info(f"skip {path_to_modulename(path)}")
            continue
        logger.info(f"generate prompt for {path_to_modulename(path)}")
        USER_PROMPT = f"""
        Here is the source code:
        ```python
        {path.read_text()}
        ```
        Please output the json format as the example with no other text.
        """
        prompt = llm.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT}
            ]
        )
        try:
            # Extract just the JSON content from the response
            content = prompt.choices[0].message.content
            if content:
                # Find the start of JSON (first '{') and end (last '}')
                json_start = content.find('```json')
                json_end = content.rfind('```')
                if json_start != -1 and json_end != -1:
                    content = content[json_start+len('```json'):json_end]
                prompt.choices[0].message.content = content
        except Exception as e:
            logger.error(f"Failed to extract JSON: {e}")
        with open(output_dir / f"{path_to_modulename(path)}.json", "w") as f:
            assert prompt.choices[0].message.content is not None
            f.write(prompt.choices[0].message.content)
        
    logger.info(f"Not found: {not_found}")

if __name__ == "__main__":
    main()
