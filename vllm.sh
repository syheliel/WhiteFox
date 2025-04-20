#!/usr/bin/bash
# vllm serve Qwen/Qwen2.5-Coder-32B-Instruct
# LOCAL_M_P=$(realpath ./llm_model/Qwen/Qwen2.5-Coder-32B-Instruct)
LOCAL_M_P=$(realpath ./llm_model/AI-ModelScope/starcoder2-7b)
MODEL_NAME="qwen/qwen-2.5-coder-32b-instruct"
echo "LOCAL MODEL PATH: $LOCAL_M_P"
echo "MODEL NAME: $MODEL_NAME"
python -m vllm.entrypoints.openai.api_server --model "$LOCAL_M_P" --served-model-name $MODEL_NAME --tensor-parallel-size 2 --gpu_memory_utilization 0.8
