#!/usr/bin/env bash
set -euo pipefail

# Legacy script (not used by the GenomeData2Vec pipeline). Kept for reference.

: "${PYTHON:=python}"

: "${SCRIPT_DIR:=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"

# Default arguments
OUT_DIM=8192 #original 65536 too large, difficult to converge, 8192 gives much better pretraining results
NORM_LAST_LAYER=True
MOMENTUM_TEACHER=0.996
USE_BN_IN_HEAD=False
WARMUP_TEACHER_TEMP=0.04
TEACHER_TEMP=0.04
WARMUP_TEACHER_TEMP_EPOCHS=0
USE_FP16=True
WEIGHT_DECAY=0.04
WEIGHT_DECAY_END=0.4
CLIP_GRAD=3.0
BATCH_SIZE_PER_GPU=1
EPOCHS=10
FREEZE_LAST_LAYER=1
LR=0.0005
WARMUP_EPOCHS=1
MIN_LR=1e-6
OPTIMIZER='adamw'
LOCAL_CROPS_NUMBER=8
GLOBAL_CROPS_SCALE=0.95
LOCAL_CROPS_SCALE=0.5
DATA_PATH='/m-ent1/ent1/xuhw/TCGA-embed/GigaPath-giant-1B/h5_files/' #'/homes/gws/jbshen/remote_files/h5_files'
EXCLUDE_DATA_PATH='./datasets/LUAD-5-gene_TCGA.csv'
OUTPUT_DIR='./test'
SAVECKP_FREQ=1
SEED=0
NUM_WORKERS=10
DIST_URL='env://'
LOCAL_RANK=0
TILE_SIZE=256
MAX_WSI_SIZE=32768
MAX_TILES=8192
SHUFFLE_TILES=False #True, False, set in params.py. Set as False gives better pretraining results
INPUT_DIM=1536
LATENT_DIM=768
GLOBAL_POOL=False
GC=32
DROPOUT=0.1
DROP_PATH_RATE=0.0
"${PYTHON}" -u "${SCRIPT_DIR}/main.py" \
    --out_dim $OUT_DIM \
    --norm_last_layer $NORM_LAST_LAYER \
    --momentum_teacher $MOMENTUM_TEACHER \
    --warmup_teacher_temp $WARMUP_TEACHER_TEMP \
    --teacher_temp $TEACHER_TEMP \
    --warmup_teacher_temp_epochs $WARMUP_TEACHER_TEMP_EPOCHS \
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
    --optimizer $OPTIMIZER \
    --local_crops_number $LOCAL_CROPS_NUMBER \
    --global_crops_scale $GLOBAL_CROPS_SCALE \
    --local_crops_scale $LOCAL_CROPS_SCALE \
    --data_path $DATA_PATH \
    --exlude_data_path $EXCLUDE_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --saveckp_freq $SAVECKP_FREQ \
    --seed $SEED \
    --num_workers $NUM_WORKERS \
    --dist_url $DIST_URL \
    --local_rank $LOCAL_RANK \
    --tile_size $TILE_SIZE \
    --max_wsi_size $MAX_WSI_SIZE \
    --max_tiles $MAX_TILES \
    --input_dim $INPUT_DIM \
    --latent_dim $LATENT_DIM \
    --gc $GC \
    --dropout $DROPOUT \
    --drop_path_rate $DROP_PATH_RATE
