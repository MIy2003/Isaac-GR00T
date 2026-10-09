#!/usr/bin/env bash
set -euo pipefail

# Single-node x86_64 CUDA deployment. Authenticate with hf auth login or HF_TOKEN.
# Inherited CUDA_VISIBLE_DEVICES is honored (including scheduler allocation).
export HF_BUNDLE_REPO="${HF_BUNDLE_REPO:-mimiclite-chip/GR00T-CHIP}"
export HF_BUNDLE_REVISION="${HF_BUNDLE_REVISION:-main}"
export DEPLOY_ROOT="${DEPLOY_ROOT:-$PWD/gr00t-chip-deployment}"
mkdir -p "$DEPLOY_ROOT"
DEPLOY_ROOT="$(cd "$DEPLOY_ROOT" && pwd)"
export DEPLOY_ROOT
DEPLOY_MODE="${DEPLOY_MODE:-train}"
case "$DEPLOY_MODE" in prepare|smoke|train) ;; *) echo "DEPLOY_MODE must be prepare, smoke or train" >&2; exit 2 ;; esac
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "This installer targets Linux x86_64 CUDA servers." >&2
  exit 2
fi
export TMPDIR="${TMPDIR:-$DEPLOY_ROOT/tmp}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$DEPLOY_ROOT/cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$DEPLOY_ROOT/python}"
export HF_HUB_CACHE="$DEPLOY_ROOT/cache/huggingface/hub"
export HF_XET_CACHE="$DEPLOY_ROOT/cache/huggingface/xet"
export TORCH_HOME="$DEPLOY_ROOT/cache/torch"
export TORCH_EXTENSIONS_DIR="$DEPLOY_ROOT/cache/torch_extensions"
export TRITON_CACHE_DIR="$DEPLOY_ROOT/cache/triton"
export CUDA_CACHE_PATH="$DEPLOY_ROOT/cache/cuda"
export MPLCONFIGDIR="$DEPLOY_ROOT/cache/matplotlib"
export HF_DATASETS_CACHE="$DEPLOY_ROOT/cache/huggingface/datasets"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$HF_HUB_CACHE" "$HF_XET_CACHE" \
  "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" \
  "$MPLCONFIGDIR" "$HF_DATASETS_CACHE" "$DEPLOY_ROOT/bootstrap"
if [[ -n "${UV_BIN:-}" ]]; then
  CHIP_UV="$UV_BIN"
elif command -v uv >/dev/null 2>&1; then
  CHIP_UV="$(command -v uv)"
else
  curl --fail --location --retry 3 https://astral.sh/uv/0.12.10/install.sh \
    --output "$DEPLOY_ROOT/bootstrap/install-uv.sh"
  UV_INSTALL_DIR="$DEPLOY_ROOT/tools" UV_NO_MODIFY_PATH=1 sh "$DEPLOY_ROOT/bootstrap/install-uv.sh"
  CHIP_UV="$DEPLOY_ROOT/tools/uv"
fi

# Resolve main once: helper, manifest and every asset use the same commit.
"$CHIP_UV" run --no-project --python 3.12 --with huggingface-hub==0.36.2 python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
repo = os.environ["HF_BUNDLE_REPO"]
revision = HfApi().repo_info(repo, revision=os.environ["HF_BUNDLE_REVISION"]).sha
destination = Path(os.environ["DEPLOY_ROOT"]) / "bootstrap"
hf_hub_download(repo, "restore_hf_bundle.py", revision=revision, local_dir=destination)
(destination / "revision.txt").write_text(revision)
print(f"Deploying {repo}@{revision}", flush=True)
PY
"$CHIP_UV" run --no-project --python 3.12 --with huggingface-hub==0.36.2 \
  python "$DEPLOY_ROOT/bootstrap/restore_hf_bundle.py" \
  --repo-id "$HF_BUNDLE_REPO" --revision "$(cat "$DEPLOY_ROOT/bootstrap/revision.txt")" \
  --destination "$DEPLOY_ROOT"
source "$DEPLOY_ROOT/deployment.env"
cd "$CHIP_PROJECT_ROOT"
"$CHIP_UV" sync --frozen --python 3.12
source .venv/bin/activate
python -m examples.CHIP.inspect_dataset --dataset-path "$CHIP_DATASET_ROOT"
if [[ "$DEPLOY_MODE" == prepare ]]; then
  echo "Prepared successfully. Project: $CHIP_PROJECT_ROOT"
  echo "Environment: $DEPLOY_ROOT/deployment.env"
  exit 0
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NUM_GPUS="${NUM_GPUS:-4}"
export OUTPUT_DIR="${OUTPUT_DIR:-$DEPLOY_ROOT/outputs/chip_${DEPLOY_MODE}}"
EXTRA_ARGS=()
if [[ "$DEPLOY_MODE" == smoke ]]; then
  EXTRA_ARGS+=(--max-steps 10 --save-steps 10 --save-total-limit 1)
fi
exec bash examples/CHIP/finetune.sh "${EXTRA_ARGS[@]}" "$@"
