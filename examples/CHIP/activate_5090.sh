#!/usr/bin/env bash
# Source this file on 5090 before training or running tests.
CHIP_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$CHIP_REPO_ROOT/.venv/bin/activate"
export GROOT_CACHE_ROOT="${GROOT_CACHE_ROOT:-/media/raid/workspace/yangmin/.cache/gr00t}"
export TMPDIR="${TMPDIR:-/media/raid/workspace/yangmin/tmp/gr00t}"
export UV_CACHE_DIR="$GROOT_CACHE_ROOT/uv"
# Keep HF_HOME unchanged so an existing hf auth login remains available.
export HF_HUB_CACHE="$GROOT_CACHE_ROOT/huggingface/hub"
export HF_XET_CACHE="$GROOT_CACHE_ROOT/huggingface/xet"
export HF_DATASETS_CACHE="$GROOT_CACHE_ROOT/huggingface/datasets"
export TORCH_HOME="$GROOT_CACHE_ROOT/torch"
export TORCH_EXTENSIONS_DIR="$GROOT_CACHE_ROOT/torch_extensions"
export TRITON_CACHE_DIR="$GROOT_CACHE_ROOT/triton"
export CUDA_CACHE_PATH="$GROOT_CACHE_ROOT/cuda"
export MPLCONFIGDIR="$GROOT_CACHE_ROOT/matplotlib"
export TEST_CACHE_PATH="$GROOT_CACHE_ROOT/tests"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$HF_HUB_CACHE" "$HF_XET_CACHE" \
  "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" \
  "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$TEST_CACHE_PATH"
