#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_PATH="${ROOT_PATH:-Time-MMD}"
OUT_DIR="${OUT_DIR:-outputs/main}"
MAX_JOBS="${MAX_JOBS:-1}"
CUDA_DEVICES="${CUDA_DEVICES:-cuda:0}"
ENABLE_WANDB="${ENABLE_WANDB:-0}"
ANCHOR_SWEEP="${ANCHOR_SWEEP:-1}"
ANCHOR_METRIC="${ANCHOR_METRIC:-MSE}"

read -r -a DEVICES <<< "${CUDA_DEVICES}"
[[ "${#DEVICES[@]}" -gt 0 ]] || { echo "CUDA_DEVICES is empty" >&2; exit 1; }

TASKS=(
  "economy|Economy/Economy.csv|m|36|6"
  "economy|Economy/Economy.csv|m|36|12"
  "economy|Economy/Economy.csv|m|36|18"
  "energy|Energy/Energy.csv|w|96|12"
  "energy|Energy/Energy.csv|w|96|24"
  "energy|Energy/Energy.csv|w|96|48"
  "security|Security/Security.csv|m|36|6"
  "security|Security/Security.csv|m|36|12"
  "security|Security/Security.csv|m|36|18"
  "socialgood|SocialGood/SocialGood.csv|m|36|6"
  "socialgood|SocialGood/SocialGood.csv|m|36|12"
  "socialgood|SocialGood/SocialGood.csv|m|36|18"
  "traffic|Traffic/Traffic.csv|m|36|6"
  "traffic|Traffic/Traffic.csv|m|36|12"
  "traffic|Traffic/Traffic.csv|m|36|18"
)

running=0
index=0
for task in "${TASKS[@]}"; do
  IFS='|' read -r name data_path freq seq_len pred_len <<< "${task}"
  device="${DEVICES[$((index % ${#DEVICES[@]}))]}"
  index=$((index + 1))
  ROOT_PATH="${ROOT_PATH}" \
  DATA_PATH="${data_path}" \
  CONFIG="${name}_${seq_len}_${pred_len}.yaml" \
  SEQ_LEN="${seq_len}" \
  PRED_LEN="${pred_len}" \
  TEXT_LEN="${seq_len}" \
  FREQ="${freq}" \
  DEVICE_STR="${device}" \
  OUT_DIR="${OUT_DIR}" \
  EXP_NAME="${name}_${seq_len}_${pred_len}" \
  ENABLE_WANDB="${ENABLE_WANDB}" \
  WANDB_PROJECT="scenariodiff_${name}_${pred_len}" \
  ANCHOR_SWEEP="${ANCHOR_SWEEP}" \
  ANCHOR_METRIC="${ANCHOR_METRIC}" \
  bash scripts/train.sh "$@" &
  running=$((running + 1))
  if [[ "${running}" -ge "${MAX_JOBS}" ]]; then
    wait -n
    running=$((running - 1))
  fi
done
wait
