#!/usr/bin/env bash
set -Eeuo pipefail

export USE_TF="${USE_TF:-0}"

ROOT_PATH="${ROOT_PATH:-Time-MMD}"
DATA_PATH="${DATA_PATH:-Economy/Economy.csv}"
CONFIG="${CONFIG:-economy_36_18.yaml}"
SEQ_LEN="${SEQ_LEN:-36}"
PRED_LEN="${PRED_LEN:-18}"
TEXT_LEN="${TEXT_LEN:-${SEQ_LEN}}"
FREQ="${FREQ:-m}"
DEVICE_STR="${DEVICE_STR:-cuda:0}"
NSAMPLE="${NSAMPLE:-50}"
SEED="${SEED:-2025}"
WITH_INTRINSIC="${WITH_INTRINSIC:-1}"
WITH_FUTURE_HINT="${WITH_FUTURE_HINT:-1}"
MIXER_SIDE_CHANNELS="${MIXER_SIDE_CHANNELS:-32}"
TEXT_BACKEND="${TEXT_BACKEND:-bert}"
TEXT_PERTURB_RATIO="${TEXT_PERTURB_RATIO:-0}"
TEXT_PERTURB_TARGETS="${TEXT_PERTURB_TARGETS:-intrinsic,future}"
TEXT_PERTURB_SEED="${TEXT_PERTURB_SEED:-${SEED}}"
ENABLE_WANDB="${ENABLE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-scenariodiff}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_MODE="${WANDB_MODE:-offline}"
ANCHOR_SWEEP="${ANCHOR_SWEEP:-1}"
ANCHOR_EVAL_SPLIT="${ANCHOR_EVAL_SPLIT:-test}"
ANCHOR_METRIC="${ANCHOR_METRIC:-MSE}"
ANCHOR_MIN_IMPROVEMENT="${ANCHOR_MIN_IMPROVEMENT:-0.0}"
OUT_DIR="${OUT_DIR:-outputs}"
EXP_NAME="${EXP_NAME:-${DATA_PATH%%/*}_${SEQ_LEN}_${PRED_LEN}}"

args=(
  -u exe_forecasting.py
  --root_path "${ROOT_PATH}"
  --data_path "${DATA_PATH}"
  --config "${CONFIG}"
  --seq_len "${SEQ_LEN}"
  --pred_len "${PRED_LEN}"
  --text_len "${TEXT_LEN}"
  --freq "${FREQ}"
  --nsample "${NSAMPLE}"
  --device "${DEVICE_STR}"
  --seed "${SEED}"
  --mixer_side_channels "${MIXER_SIDE_CHANNELS}"
  --text_backend "${TEXT_BACKEND}"
  --text_perturb_ratio "${TEXT_PERTURB_RATIO}"
  --text_perturb_targets "${TEXT_PERTURB_TARGETS}"
  --text_perturb_seed "${TEXT_PERTURB_SEED}"
  --output_dir "${OUT_DIR}"
  --exp_name "${EXP_NAME}"
)

[[ "${WITH_INTRINSIC}" == "1" ]] && args+=(--with_intrinsic)
[[ "${WITH_FUTURE_HINT}" == "1" ]] && args+=(--with_future_hint)
[[ "${ENABLE_WANDB}" == "1" ]] && args+=(--enable_wandb --wandb_project "${WANDB_PROJECT}" --wandb_mode "${WANDB_MODE}")
[[ -n "${WANDB_GROUP}" ]] && args+=(--wandb_group "${WANDB_GROUP}")
[[ "${ANCHOR_SWEEP}" == "1" ]] && args+=(--anchor_sweep --anchor_eval_split "${ANCHOR_EVAL_SPLIT}" --anchor_metric "${ANCHOR_METRIC}" --anchor_min_improvement "${ANCHOR_MIN_IMPROVEMENT}")

python "${args[@]}" "$@"
