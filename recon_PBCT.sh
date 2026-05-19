#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate pytorch

TASK="SVCT"
DEGREE=20

GPU=0

SAVE_DIR="./results/"

CHECKPOINT_PATH="./checkpoints/edm/network-snapshot-003882.pkl"

DATA="./data/AbdomenCT-1K/valid/Case_00066_0000.nii.gz"

SLICE_BEGIN=0
SLICE_END=500
SLICE_STEP=10

RECON_SIZE=256

NOISE_CONTROL=None
USE_INIT=True
RENOISE_METHOD=DDPM
SIGMA_MIN=0.01
SIGMA_MAX=2

NFE=50
NUM_CG=50
W_TIK=0

METHOD=DiffPIR
# METHOD=DCPnPDP

echo "============================================================"
echo "Run PBCT reconstruction"
echo "METHOD=${METHOD}, TASK=${TASK}, DEGREE=${DEGREE}, DATA=$(basename "${DATA}")"
echo "SLICE=${SLICE_BEGIN}:${SLICE_END}:${SLICE_STEP}, RECON_SIZE=${RECON_SIZE}"
echo "NFE=${NFE}, NUM_CG=${NUM_CG}, SIGMA_MIN=${SIGMA_MIN}, SIGMA_MAX=${SIGMA_MAX}"
if [ "${METHOD}" = "DAPS" ]; then
    echo "DAPS_DIFFUSION_NUM_STEPS=${DAPS_DIFFUSION_NUM_STEPS}, DAPS_DIFFUSION_SIGMA_MIN=${DAPS_DIFFUSION_SIGMA_MIN}, DAPS_LGVD_NUM_STEPS=${DAPS_LGVD_NUM_STEPS}, DAPS_LGVD_LR=${DAPS_LGVD_LR}, DAPS_LGVD_TAU=${DAPS_LGVD_TAU}, DAPS_LGVD_LR_MIN_RATIO=${DAPS_LGVD_LR_MIN_RATIO}, DAPS_DENOISE_BATCH_SIZE=${DAPS_DENOISE_BATCH_SIZE}"
fi
echo "============================================================"


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
--sigma-min $SIGMA_MIN \
--renoise-method $RENOISE_METHOD \
--checkpoint-path $CHECKPOINT_PATH \
--save_dir $SAVE_DIR
