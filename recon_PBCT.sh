#!/bin/bash

source /mnt/sda/dch/code/DPER/25-09-ICLR-ISCS/auto_gpu_select.sh

NUM_GPUS_TO_USE=1
ALL_FREE_GPUS_STR=$(find_free_gpus $NUM_GPUS_TO_USE)

echo "Found free GPUs: ${ALL_FREE_GPUS_STR}"

export CUDA_VISIBLE_DEVICES=${ALL_FREE_GPUS_STR}

echo "Successfully set CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"


CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate pytorch

TASK="SVCT"
DEGREE=20

# TASK="LACT"
# DEGREE=90

GPU=0

SAVE_DIR="./results/"

# CHECKPOINT_PATH="./checkpoints/edm/network-snapshot-003882.pkl"
CHECKPOINT_PATH="/mnt/MIXVol-1/dch/checkpoint/edm/00010-AbdomenCT1K-uncond-ddpmpp-edm-gpus8-batch112-fp32/network-snapshot-003882.pkl"

# DATA="./data/AbdomenCT-1K/valid/Case_00066_0000.nii.gz"
DATA="/mnt/MIXVol-1/dch/data/AbdomenCT-1K/selected/valid/gantry_removed/Case_00066_0000.nii.gz"


SLICE_BEGIN=0
SLICE_END=500
SLICE_STEP=100

RECON_SIZE=256

NOISE_CONTROL=None
USE_INIT=True
RENOISE_METHOD=DDPM
SIGMA_MAX=2

NFE=50
NUM_CG=10
W_TIK=0

# METHOD=DiffPIR
METHOD=DCPnPDP
# METHOD=SITCOM



python recon_PBCT.py \
--method $METHOD \
--task $TASK \
--degree $DEGREE \
--gpu $GPU \
--data $DATA \
--slice-begin $SLICE_BEGIN \
--slice-end $SLICE_END \
--slice-step $SLICE_STEP \
--recon-size $RECON_SIZE \
--NFE $NFE \
--num-cg $NUM_CG \
--w-tik $W_TIK \
--noise-control $NOISE_CONTROL \
--use-init $USE_INIT \
--sigma-max $SIGMA_MAX \
--renoise-method $RENOISE_METHOD \
--checkpoint-path $CHECKPOINT_PATH \
--save_dir $SAVE_DIR

