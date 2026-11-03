#!/usr/bin/env bash
#SBATCH --job-name=chip-ta-force
#SBATCH --partition=DGX,HGX
#SBATCH --account=research
#SBATCH --qos=lv0b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=ta-force-%j.out
#SBATCH --error=ta-force-%j.err
set -eo pipefail
source /etc/profile
set -u
export DEPLOY_ROOT=/share/ml/yangmin/gr00t-chip
source "$DEPLOY_ROOT/deployment.env"
: "${TA_PROJECT_ROOT:?Set the isolated code snapshot}"
: "${TA_TASK:?Use dough or pivot}"
case "$TA_TASK" in
  dough)
    dataset="$DEPLOY_ROOT/data/dough_shaving_replay_merged_v001"
    base="$DEPLOY_ROOT/outputs/dough_shaving_replay_yaw_1gpu_5284/dough_shaving_replay_yaw_1gpu/checkpoint-20000"
    task='Shave the dough.' ;;
  pivot)
    dataset="$DEPLOY_ROOT/data/pivot_box_replay_merged_v002"
    base="$DEPLOY_ROOT/outputs/pivot_box_replay_yaw_1gpu_5448/pivot_box_replay_yaw_1gpu/checkpoint-20000"
    task='Pivot the box.' ;;
  *) echo "Unknown TA_TASK: $TA_TASK" >&2; exit 1 ;;
esac
export CUDA_HOME=/opt/ohpc/pub/apps/cuda12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$TA_PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH="$TA_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export NO_ALBUMENTATIONS_UPDATE=1 JE_ARROW_MALLOC_CONF=background_thread:false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$(mktemp -d /tmp/gr00t-ta-force-${SLURM_JOB_ID}.XXXXXX)"
export TORCH_HOME="$DEPLOY_ROOT/cache/torch"
export TORCH_EXTENSIONS_DIR="$DEPLOY_ROOT/cache/torch_extensions"
export TRITON_CACHE_DIR="$DEPLOY_ROOT/cache/triton"
export CUDA_CACHE_PATH="$DEPLOY_ROOT/cache/cuda"
export MPLCONFIGDIR="$DEPLOY_ROOT/cache/matplotlib"
trap 'rm -rf -- "$TMPDIR"' EXIT
base_config="$base/experiment_cfg/config.yaml"
srun --ntasks=1 --kill-on-bad-exit=1 python -m examples.CHIP.train_ta_force \
  --base-config "$base_config" \
  --output-dir "$DEPLOY_ROOT/outputs/${TA_TASK}_ta_force_xyz_rtc_base_${SLURM_JOB_ID}"
