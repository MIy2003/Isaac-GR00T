#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Activate the Isaac-GR00T Python 3.12 environment before invoking this script.
: "${CHIP_DATASET_ROOT:?Set CHIP_DATASET_ROOT to the folder containing dataset_manifest.json}"
NUM_GPUS="${NUM_GPUS:-1}"
# Keep torchrun's world size and the training configuration in sync, including
# when callers use the same --num-gpus override as the upstream launcher.
EXTRA_ARGS=()
while (($#)); do
  case "$1" in
    --num-gpus|--num_gpus)
      NUM_GPUS="${2:?--num-gpus requires a value}"
      shift 2 ;;
    --num-gpus=*|--num_gpus=*)
      NUM_GPUS="${1#*=}"
      shift ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done
if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS must be a positive integer" >&2
  exit 2
fi
LAUNCH=(python)
if ((NUM_GPUS > 1)); then
  LAUNCH=(torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS")
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi
exec "${LAUNCH[@]}" gr00t/experiment/launch_finetune.py \
  --base-model-path "${BASE_MODEL_PATH:-nvidia/GR00T-N1.7-3B}" \
  --dataset-path "$CHIP_DATASET_ROOT" \
  --dataset-format "${CHIP_DATASET_FORMAT:-chip_native}" \
  --embodiment-tag NEW_EMBODIMENT \
  --task-description "${CHIP_TASK_DESCRIPTION:-Perform the demonstrated task.}" \
  --output-dir "${OUTPUT_DIR:-./outputs/chip_native_relative}" \
  --num-gpus "$NUM_GPUS" \
  --global-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --dataloader-num-workers 2 \
  --shard-size 128 \
  --state-dropout-prob 0 \
  --rtc-training-max-delay "${RTC_MAX_DELAY:-6}" \
  --no-use-percentiles \
  --shortest-image-edge 224 \
  --crop-fraction 1.0 \
  --max-steps 10000 \
  --save-steps 1000 \
  "${EXTRA_ARGS[@]}"
