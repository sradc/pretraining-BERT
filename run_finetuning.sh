#!/bin/bash
set -e

RUN_PATH="<path to run directory>"  # <- you need to change this
mkdir "${RUN_PATH}/glue"

for TASK_NAME in stsb mnli rte qqp mrpc cola sst2 qnli
do
    export WANDB_PROJECT="BERT-${TASK_NAME}"
    printf "$(date) starting ${TASK_NAME}\n"
    python glue_eval.py \
        --model_name_or_path "${RUN_PATH}/model" \
        --task_name "${TASK_NAME}" \
        --do_train \
        --do_eval \
        --do_predict \
        --max_seq_length 124 \
        --per_device_train_batch_size 32 \
        --learning_rate 2e-5 \
        --lr_scheduler_type linear \
        --num_train_epochs 5 \
        --save_strategy no  \
        --seed 42  \
        --output_dir "${RUN_PATH}/glue/${TASK_NAME}"
    printf "$(date) finished ${TASK_NAME}\n\n"
done
