#! /bin/bash
set -e
INPUT_DIR=$(realpath ~/pytorch/torch)
echo "INPUT_DIR: $INPUT_DIR"
poetry run python -m src.db.build_db --batch-size 40 --num-workers 8  --extensions "rst,py" --embedding-type huggingface --input-dir $INPUT_DIR --num-workers 8

