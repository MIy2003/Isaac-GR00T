#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Activate this machine's GR00T Python 3.12 environment before running.
# A100 80 GB: 128 samples per GPU, four GPUs, accumulation 1.
# Use a smaller GLOBAL_BATCH_SIZE on 40 GB GPUs.
: "${CHIP_DATASET_ROOT:?Set CHIP_DATASET_ROOT to the folder containing dataset_manifest.json}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OUTPUT_DIR="${OUTPUT_DIR:-./outputs/chip_native_relative_a100_4gpu}"

# Preserve learning rate, frozen backbone, CHIP horizon and RTC settings.
# GLOBAL_BATCH_SIZE is summed over all four GPUs BEFORE accumulation.
# Keep the generic launcher as the single owner of torchrun and CHIP I/O flags.
exec bash examples/CHIP/finetune.sh \
  --num-gpus 4 \
  --global-batch-size "${GLOBAL_BATCH_SIZE:-512}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --learning-rate "${LEARNING_RATE:-1e-4}" \
  --no-tune-llm \
  --no-tune-visual \
  --tune-projector \
  --tune-diffusion-model \
  --max-steps "${MAX_STEPS:-10000}" \
  --save-steps "${SAVE_STEPS:-1000}" \
  "$@"
