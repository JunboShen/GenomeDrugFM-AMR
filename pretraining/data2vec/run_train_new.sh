#!/usr/bin/env bash
set -euo pipefail

# GenomeData2Vec pretraining (paper: GenomeDrugFM-AMR).
#
# Override any of these via environment variables, e.g.:
#   CUDA_VISIBLE_DEVICES=0 OUTPUT_DIR=./outputs/pretrain bash pretraining/data2vec/run_train_new.sh

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

: "${PYTHON:=python}"

: "${SCRIPT_DIR:=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"

: "${TRAIN_CSV:=./data/PATRIC_genomes_AMR_pretrain.csv}"
: "${ONEHOT_DIR:=./one_hots_data2vec_new}"
: "${SPLIT_INDICES_DIR:=./one_hots_split_indices_data2vec_new}"

: "${PATCH_SIZE:=4096}"
: "${EMBED_DIM:=1536}"

: "${REPORT_TO:=wandb}"           # wandb|none
: "${WANDB_PROJECT:=genomeData2Vec}"
: "${WANDB_MODE:=online}"         # online|offline|disabled

: "${NORM_LAST_LAYER:=True}"
: "${USE_FP16:=True}"
: "${WEIGHT_DECAY:=0.04}"
: "${WEIGHT_DECAY_END:=0.4}"
: "${CLIP_GRAD:=3.0}"
: "${BATCH_SIZE_PER_GPU:=2}"
: "${EPOCHS:=20}"
: "${FREEZE_LAST_LAYER:=1}"
: "${LR:=0.0005}"
: "${WARMUP_EPOCHS:=2}"
: "${MIN_LR:=1e-6}"
: "${OUTPUT_DIR:=./outputs/pretrain_patch4096}"
: "${SAVECKP_FREQ:=2}"
: "${SEED:=0}"
: "${NUM_WORKERS:=8}"
: "${GC:=64}"
: "${DROPOUT:=0.1}"
: "${DROP_PATH_RATE:=0.0}"

 "${PYTHON}" -u "${SCRIPT_DIR}/train_new.py" \
  --train_csv "${TRAIN_CSV}" \
  --onehot_dir "${ONEHOT_DIR}" \
  --split_indices_dir "${SPLIT_INDICES_DIR}" \
  --patch_size "${PATCH_SIZE}" \
  --embed_dim "${EMBED_DIM}" \
  --report_to "${REPORT_TO}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_mode "${WANDB_MODE}" \
  --norm_last_layer "${NORM_LAST_LAYER}" \
  --use_fp16 "${USE_FP16}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --weight_decay_end "${WEIGHT_DECAY_END}" \
  --clip_grad "${CLIP_GRAD}" \
  --batch_size_per_gpu "${BATCH_SIZE_PER_GPU}" \
  --epochs "${EPOCHS}" \
  --freeze_last_layer "${FREEZE_LAST_LAYER}" \
  --lr "${LR}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --min_lr "${MIN_LR}" \
  --output_dir "${OUTPUT_DIR}" \
  --saveckp_freq "${SAVECKP_FREQ}" \
  --seed "${SEED}" \
  --num_workers "${NUM_WORKERS}" \
  --gc "${GC}" \
  --dropout "${DROPOUT}" \
  --drop_path_rate "${DROP_PATH_RATE}"
