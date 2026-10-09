#!/usr/bin/env bash
#SBATCH --job-name=gr00t-env
#SBATCH --partition=DGX
#SBATCH --account=research
#SBATCH --qos=lv0b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00

set -euo pipefail
source /etc/profile
module add cuda12.8/12.8
export DEPLOY_ROOT="${DEPLOY_ROOT:-/share/ml/yangmin/gr00t-chip}"
source "$DEPLOY_ROOT/deployment.env"
source "$CHIP_PROJECT_ROOT/.venv/bin/activate"
export CUDA_VISIBLE_DEVICES=""
export UV_CACHE_DIR="$DEPLOY_ROOT/cache/uv" TMPDIR="$DEPLOY_ROOT/tmp"
export FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=80
export TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=8 NVCC_THREADS=2
export NO_ALBUMENTATIONS_UPDATE=1 OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
hostname
getconf GNU_LIBC_VERSION
cd "$(cat "$DEPLOY_ROOT/flash-build-source.txt")"
python setup.py bdist_wheel --dist-dir "$DEPLOY_ROOT/wheels"
"$DEPLOY_ROOT/tools/uv" pip install --python "$VIRTUAL_ENV/bin/python" \
  --no-deps --reinstall-package flash-attn \
  "$DEPLOY_ROOT/wheels/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"
cd "$CHIP_PROJECT_ROOT"
export PYTHONPATH="$CHIP_PROJECT_ROOT"
python /home/yangmin/gr00t-bootstrap/validate_prepared_env.py
