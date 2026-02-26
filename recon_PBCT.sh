#!/bin/bash


export CUDA_VISIBLE_DEVICES=0


CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate pytorch

TASK="SVCT"
DEGREE=20

GPU=0

SAVE_DIR="./results/"

CHECKPOINT_PATH="./checkpoint/edm/network-snapshot-003882.pkl"

DATA="./data/AbdomenCT-1K/valid/Case_00066_0000.nii.gz"


SLICE_BEGIN=0
SLICE_END=500
SLICE_STEP=10

RECON_SIZE=256

NOISE_CONTROL=None
USE_INIT=True
RENOISE_METHOD=DDPM
SIGMA_MAX=2

NFE=50
NUM_CG=50
W_TIK=0

METHOD=DiffPIR
# METHOD=DCPnPDP



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

