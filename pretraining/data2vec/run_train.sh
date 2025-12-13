#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

: "${PYTHON:=python}"

: "${SCRIPT_DIR:=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"

# Legacy pretraining entrypoint (kept for reference).
# Default arguments
NORM_LAST_LAYER=True
USE_FP16=True
WEIGHT_DECAY=0.04
WEIGHT_DECAY_END=0.4
CLIP_GRAD=3.0
BATCH_SIZE_PER_GPU=1
EPOCHS=5
FREEZE_LAST_LAYER=1
LR=0.0005
WARMUP_EPOCHS=0
MIN_LR=1e-6
OUTPUT_DIR='./test8192' #'./test2048' #'./test8192' #'./test2'
SAVECKP_FREQ=1
SEED=0
NUM_WORKERS=1
GC=32 #original 32
DROPOUT=0.1
DROP_PATH_RATE=0.0
"${PYTHON}" -u "${SCRIPT_DIR}/train.py" \
    --norm_last_layer $NORM_LAST_LAYER \
    --use_fp16 $USE_FP16 \
    --weight_decay $WEIGHT_DECAY \
    --weight_decay_end $WEIGHT_DECAY_END \
    --clip_grad $CLIP_GRAD \
    --batch_size_per_gpu $BATCH_SIZE_PER_GPU \
    --epochs $EPOCHS \
    --freeze_last_layer $FREEZE_LAST_LAYER \
    --lr $LR \
    --warmup_epochs $WARMUP_EPOCHS \
    --min_lr $MIN_LR \
    --output_dir $OUTPUT_DIR \
    --saveckp_freq $SAVECKP_FREQ \
    --seed $SEED \
    --num_workers $NUM_WORKERS \
    --gc $GC \
    --dropout $DROPOUT \
    --drop_path_rate $DROP_PATH_RATE
