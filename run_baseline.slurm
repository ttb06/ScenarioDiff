#!/bin/bash
#SBATCH --job-name=run_single
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=outputs_slurm/ene_with_anchor_%j.out
#SBATCH --error=outputs_slurm/ene_with_anchor_%j.out

# bash scripts/agriculture.sh
# bash scripts/climate.sh

export PYTHONNOUSERSITE=1

export USE_TF=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ROOT_PATH=Time-MMD \
DATA_PATH=Economy/Economy.csv \
CONFIG=economy_36_18.yaml \
SEQ_LEN=36 PRED_LEN=18 TEXT_LEN=36 FREQ=m \
bash scripts/train.sh