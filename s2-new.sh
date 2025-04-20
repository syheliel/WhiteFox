set -e
TARGET="torch-inductor"
PROMPT_3_DIR="./prompt-3/${TARGET}"
GENCODE_DIR="./code-3-new"
# GENCODE_DIR="success/"
RESULT_DIR="./result-4-new"
# MODEL="qw" # 火山引擎

# python gpt4_gencode.py --prompt-dir=${PROMPT_3_DIR} --output-dir=${GENCODE_DIR} --num=5 --model=${MODEL}
poetry run exec_code --input-dir=${GENCODE_DIR} --result-dir=${RESULT_DIR} --cov
