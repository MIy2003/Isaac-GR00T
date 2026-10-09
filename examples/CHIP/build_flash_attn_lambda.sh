#!/usr/bin/env bash
# The upstream cp312 wheel requires GLIBC_2.32; lambda runs Ubuntu 20.04.
# Build the same release against lambda's libc and only the A100 architecture.
set -euo pipefail
source /etc/profile
module add cuda12.8/12.8
export DEPLOY_ROOT="${DEPLOY_ROOT:-/share/ml/yangmin/gr00t-chip}"
source "$DEPLOY_ROOT/deployment.env"
cd "$CHIP_PROJECT_ROOT"
source .venv/bin/activate
export UV_CACHE_DIR="$DEPLOY_ROOT/cache/uv" TMPDIR="$DEPLOY_ROOT/tmp"
export UV_HTTP_TIMEOUT=180 UV_HTTP_RETRIES=5
"$DEPLOY_ROOT/tools/uv" pip install --python "$VIRTUAL_ENV/bin/python" wheel ninja packaging setuptools
export FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=80
export TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=4 NVCC_THREADS=2
nice -n 10 "$DEPLOY_ROOT/tools/uv" pip install --python "$VIRTUAL_ENV/bin/python" \
  --no-deps --no-build-isolation --no-binary flash-attn --reinstall-package flash-attn \
  "flash-attn==2.8.3"
