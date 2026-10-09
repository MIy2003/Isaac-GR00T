#!/usr/bin/env bash
#SBATCH --job-name=chip-sim-yaw
#SBATCH --partition=DGX,HGX
#SBATCH --account=research
#SBATCH --qos=lv0b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=chip-sim-yaw-%j.out
#SBATCH --error=chip-sim-yaw-%j.err
set -eo pipefail
source /etc/profile
set -u
# The current cluster provides Slurm directly and does not initialize Lmod
# in batch shells. Use the installed CUDA toolkit explicitly.
export CUDA_HOME=/opt/ohpc/pub/apps/cuda12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
: "${SLURM_JOB_ID:?Submit with sbatch}"
export DEPLOY_ROOT="${DEPLOY_ROOT:-/share/ml/yangmin/gr00t-chip}"
source "$DEPLOY_ROOT/deployment.env"
export CHIP_PROJECT_ROOT="${REPLAY_PROJECT_ROOT:-$DEPLOY_ROOT/runs/wiping_car_sim_replay_yaw_20261006/Isaac-GR00T}"
export CHIP_DATASET_ROOT="${REPLAY_DATASET_ROOT:-$DEPLOY_ROOT/data/wiping_car_sim_replay_862_c005_v001}"
cd "$CHIP_PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH="$CHIP_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CHIP_DATASET_FORMAT=chip_replay_yaw_canonical
export CHIP_TASK_DESCRIPTION='Wipe the car.'
export NUM_GPUS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export NO_ALBUMENTATIONS_UPDATE=1 JE_ARROW_MALLOC_CONF=background_thread:false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OUTPUT_DIR="${OUTPUT_DIR:-$DEPLOY_ROOT/outputs/wiping_car_sim_replay_yaw_1gpu_${SLURM_JOB_ID}}"
export TMPDIR="$(mktemp -d /tmp/gr00t-replay-${SLURM_JOB_ID}.XXXXXX)"
export TORCH_HOME="$DEPLOY_ROOT/cache/torch"
export TORCH_EXTENSIONS_DIR="$DEPLOY_ROOT/cache/torch_extensions"
export TRITON_CACHE_DIR="$DEPLOY_ROOT/cache/triton"
export CUDA_CACHE_PATH="$DEPLOY_ROOT/cache/cuda"
export MPLCONFIGDIR="$DEPLOY_ROOT/cache/matplotlib"
trap 'rm -rf -- "$TMPDIR"' EXIT
python -m examples.CHIP.inspect_replay_dataset --dataset-path "$CHIP_DATASET_ROOT" --yaw-canonical
srun --ntasks=1 --kill-on-bad-exit=1 bash examples/CHIP/finetune.sh \
  --experiment-name wiping_car_sim_replay_yaw_1gpu \
  --global-batch-size "${GLOBAL_BATCH_SIZE:-160}" \
  --gradient-accumulation-steps 1 \
  --no-tune-llm --no-tune-visual --tune-projector --tune-diffusion-model \
  --max-steps "${MAX_STEPS:-20000}" --save-steps "${SAVE_STEPS:-1000}" "$@"
