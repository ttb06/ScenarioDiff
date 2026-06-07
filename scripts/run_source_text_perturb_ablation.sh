#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_PATH="${ROOT_PATH:-Time-MMD}"
OUT_DIR="${OUT_DIR:-outputs/source_text_perturb}"
MAX_JOBS="${MAX_JOBS:-1}"
CUDA_DEVICES="${CUDA_DEVICES:-cuda:0}"
RATIOS="${RATIOS:-0.10 0.20 0.30 0.40 0.50}"
TEXT_PERTURB_SEED="${TEXT_PERTURB_SEED:-2025}"

read -r -a DEVICES <<< "${CUDA_DEVICES}"
read -r -a RATIO_LIST <<< "${RATIOS}"
TASKS=(
  "economy|Economy/Economy.csv|m|36|18"
  "energy|Energy/Energy.csv|w|96|48"
  "security|Security/Security.csv|m|36|18"
  "socialgood|SocialGood/SocialGood.csv|m|36|18"
  "traffic|Traffic/Traffic.csv|m|36|18"
)

running=0
index=0
for task in "${TASKS[@]}"; do
  IFS='|' read -r name data_path freq seq_len pred_len <<< "${task}"
  for ratio in "${RATIO_LIST[@]}"; do
    device="${DEVICES[$((index % ${#DEVICES[@]}))]}"
    index=$((index + 1))
    ROOT_PATH="${ROOT_PATH}" \
    DATA_PATH="${data_path}" \
    CONFIG="${name}_${seq_len}_${pred_len}.yaml" \
    SEQ_LEN="${seq_len}" PRED_LEN="${pred_len}" TEXT_LEN="${seq_len}" FREQ="${freq}" \
    DEVICE_STR="${device}" OUT_DIR="${OUT_DIR}" \
    EXP_NAME="${name}_${seq_len}_${pred_len}_text_${ratio}" \
    TEXT_PERTURB_RATIO="${ratio}" TEXT_PERTURB_SEED="${TEXT_PERTURB_SEED}" \
    ANCHOR_SWEEP=0 bash scripts/train.sh "$@" &
    running=$((running + 1))
    if [[ "${running}" -ge "${MAX_JOBS}" ]]; then
      wait -n
      running=$((running - 1))
    fi
  done
done
wait
python tools/collect_text_perturb_ablation.py --root "${OUT_DIR}" --metric MSE
