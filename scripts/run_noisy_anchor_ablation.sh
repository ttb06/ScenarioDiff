#!/usr/bin/env bash
set -Eeuo pipefail

[[ -n "${CHECKPOINT_DIRS:-}" ]] || {
  echo "Set CHECKPOINT_DIRS to one or more trained checkpoint directories." >&2
  exit 1
}
read -r -a CHECKPOINT_LIST <<< "${CHECKPOINT_DIRS}"

python -u tools/noisy_anchor_ablation.py \
  --checkpoint_dirs "${CHECKPOINT_LIST[@]}" \
  --output_dir "${OUT_DIR:-outputs/noisy_anchor_ablation}" \
  --device "${DEVICE_STR:-cuda:0}" \
  --nsample "${NSAMPLE:-50}" \
  --metric "${METRIC:-MSE}" \
  --levels ${LEVELS:-20 30 50 100} \
  --noisy_seed "${NOISY_SEED:-2025}" \
  "$@"
