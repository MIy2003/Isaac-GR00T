#!/usr/bin/env bash
#SBATCH --job-name=train2
#SBATCH --partition=DGX,HGX
#SBATCH --account=research
#SBATCH --qos=lv0b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=train2-%j.out
#SBATCH --error=train2-%j.err

set -eo pipefail

# lambda: install with DEPLOY_MODE=prepare and run validate_prepared_env.py
# BEFORE sbatch. This job never downloads resources or installs dependencies.
source /etc/profile
set -u
module add slurm cuda12.8/12.8
: "${SLURM_JOB_ID:?Submit this file with sbatch}"
if [[ "${SLURM_JOB_NUM_NODES:-1}" != 1 || "${SLURM_NTASKS:-1}" != 1 ]]; then
  echo "Request one node and one Slurm task; torchrun starts four workers." >&2
  exit 2
fi
export DEPLOY_ROOT="${DEPLOY_ROOT:-/share/ml/yangmin/gr00t-chip}"
: "${DEPLOY_ROOT:?}"
source "$DEPLOY_ROOT/deployment.env"
cd "$CHIP_PROJECT_ROOT"
source .venv/bin/activate
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export NO_ALBUMENTATIONS_UPDATE=1
export JE_ARROW_MALLOC_CONF=background_thread:false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NUM_GPUS=4
export OUTPUT_DIR="${OUTPUT_DIR:-$DEPLOY_ROOT/outputs/a100_visual_4gpu_${SLURM_JOB_ID}}"
export TMPDIR="$(mktemp -d /tmp/gr00t-chip-${SLURM_JOB_ID}.XXXXXX)"
export TORCH_HOME="$DEPLOY_ROOT/cache/torch"
export TORCH_EXTENSIONS_DIR="$DEPLOY_ROOT/cache/torch_extensions"
export TRITON_CACHE_DIR="$DEPLOY_ROOT/cache/triton"
export CUDA_CACHE_PATH="$DEPLOY_ROOT/cache/cuda"
export MPLCONFIGDIR="$DEPLOY_ROOT/cache/matplotlib"
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

python - <<'PY_CHECK'
import hashlib, json, os
from pathlib import Path
root = Path(os.environ["DEPLOY_ROOT"])
report = json.loads((root / "environment-verified.json").read_text())
assert report["passed"], "Environment validation did not pass"
assert report["revision"] == os.environ["CHIP_BUNDLE_REVISION"], "Revalidate changed bundle"
assert report["deployment_env_sha256"] == hashlib.sha256((root / "deployment.env").read_bytes()).hexdigest(), "Revalidate changed environment"
assert report["project"] == os.environ["CHIP_PROJECT_ROOT"]
print("Using validated offline environment:", report["project"], flush=True)
PY_CHECK

# Train visual backbone + action head; freeze LLM. 4 x batch 24 = global 96.
exec srun --ntasks=1 --kill-on-bad-exit=1 \
  bash examples/CHIP/finetune.sh \
  --experiment-name train2 \
  --global-batch-size "${GLOBAL_BATCH_SIZE:-96}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --no-tune-llm \
  --tune-visual \
  --tune-projector \
  --tune-diffusion-model \
  --max-steps "${MAX_STEPS:-20000}" \
  --save-steps "${SAVE_STEPS:-1000}" \
  "$@"
