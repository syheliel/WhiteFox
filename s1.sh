#!/bin/bash
set -e
TARGET="torch-inductor"
PROMPT_1_DIR="./prompt-1/${TARGET}"
REQUIREMENT_2_DIR="./requirement-2/${TARGET}"
TEMPLATE_DIR="template-torch"
MODEL="deepseek-v3-250324"

for optim_name in 'inductor' 'group-batch'; do
    python prompt_gen.py --optpath=optim/inductor-${optim_name}.json --mode=src2nl --template=${TEMPLATE_DIR}/starcoder-src2nl.txt --outdir=${PROMPT_1_DIR}
done

for optim_name in 'postgrad' 'mkldnn' 'sfdp' 'bnfold' 'decompose' 'misc' 'fuse' 'quant' 'graph-pattern'; do
    python prompt_gen.py --optpath=optim/inductor-${optim_name}.json --mode=src2nl --template=${TEMPLATE_DIR}/starcoder-src2nl-pattern.txt --outdir=${PROMPT_1_DIR}
done

python gpt4.py --prompt-dir=${PROMPT_1_DIR} \
    --outdir=${REQUIREMENT_2_DIR} \
    --temperature=0.0 \
    --model=${MODEL} \
    --batch-size=5