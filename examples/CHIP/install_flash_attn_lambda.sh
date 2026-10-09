#!/usr/bin/env bash
# Community manylinux wheel compatible with lambda's glibc 2.31.
set -euo pipefail
export DEPLOY_ROOT="${DEPLOY_ROOT:-/share/ml/yangmin/gr00t-chip}"
source "$DEPLOY_ROOT/deployment.env"
python="$CHIP_PROJECT_ROOT/.venv/bin/python"
"$python" - <<'PY'
import sys, torch
assert sys.version_info[:2] == (3, 12)
assert torch.__version__.split('+')[0] == '2.9.0'
assert torch.version.cuda == '12.8' and torch._C._GLIBCXX_USE_CXX11_ABI
PY
name=flash_attn-2.8.3+cu128torch2.9-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
url="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/$name"
sha=cbf56ce702b705525aacd097506146b24eb5c76b9f3e33eb7d0b5a6ee64117f2
mkdir -p "$DEPLOY_ROOT/wheels"
wheel="$DEPLOY_ROOT/wheels/$name"
if ! echo "$sha  $wheel" | sha256sum -c - >/dev/null 2>&1; then
  curl -fL --retry 5 --connect-timeout 30 -o "$wheel.part" "$url"
  echo "$sha  $wheel.part" | sha256sum -c -
  mv "$wheel.part" "$wheel"
fi
rm -f "$DEPLOY_ROOT/environment-verified.json"
export UV_CACHE_DIR="$DEPLOY_ROOT/cache/uv"
"$DEPLOY_ROOT/tools/uv" pip install --python "$python" --no-deps \
  --reinstall-package flash-attn "$wheel"
"$python" -c 'import torch, flash_attn, flash_attn_2_cuda; print("FlashAttention native import passed:", flash_attn.__version__)'
printf '%s\nsha256:%s\n' "$url" "$sha" > "$DEPLOY_ROOT/wheels/flash-attn-source.txt"
