set -e
TARGET="torch-inductor"
PROMPT_3_DIR="./prompt-3/${TARGET}"
GENCODE_DIR="./gencode-4"
RESULT_DIR="./result-4"
MODEL="deepseek-v3-250324" # 火山引擎

python gpt4_gencode.py --prompt-dir=${PROMPT_3_DIR} --output-dir=${GENCODE_DIR} --num=5 --model=${MODEL}
python -m torch-exec.run_torch --input-dir=${GENCODE_DIR} --res-dir=${RESULT_DIR}