from typing import List
import openai
from pathlib import Path

ALL_MODEL: List[str] = [
    "gpt-4",
    "gpt-4o",
    "gpt-4o-mini",
    "deepseek-chat",
    "deepseek-coder",
    "deepseek-reasoner",
    "deepseek-v3-250324", # 火山引擎
    "qwen/qwen-2.5-coder-32b-instruct" # open router
] 
def get_openai(model:str,use_vllm:bool = False) -> openai.OpenAI:
    if model not in ALL_MODEL:
        raise ValueError(f"Model {model} is not supported")
    if use_vllm:
        base_url = "http://localhost:8000/v1"
        return openai.OpenAI(api_key="no_need_any_key", base_url = base_url)

    if model in ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"]:
        api_key = Path("deepseek.key").read_text().strip()
        base_url = "https://api.deepseek.com/v1"
    elif model in ["deepseek-v3-250324"]:
        api_key = Path("ark.key").read_text().strip()
        base_url = "https://ark.cn-beijing.volces.com/api/v3"
    elif model in ["qwen/qwen-2.5-coder-32b-instruct"]:
        api_key = Path("openrouter.key").read_text().strip()
        base_url = "https://openrouter.ai/api/v1"
    else:
        api_key = Path("openai.key").read_text().strip()
        base_url = "https://api.openai.com/v1"

    return openai.OpenAI(api_key=api_key, base_url=base_url)
