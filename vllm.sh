#!/usr/bin/bash
# vllm serve Qwen/Qwen2.5-Coder-32B-Instruct
python -m vllm.entrypoints.openai.api_server --model "./cache_dir/Qwen/Qwen2.5-32B-Instruct" --served-model-name "qwen/qwen-2.5-coder-32b-instruct" --tensor-parallel-size 2
