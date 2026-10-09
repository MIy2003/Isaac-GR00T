"""Validate a prepared CHIP environment on a CPU login node before sbatch.

Source deployment.env, activate the restored .venv, and load CUDA 12.8 first.
GPU kernels and multi-GPU communication still require an allocated GPU node.
"""

import hashlib
from importlib import import_module
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch


def main():
    root = Path(os.environ["DEPLOY_ROOT"])
    project = Path(os.environ["CHIP_PROJECT_ROOT"])
    marker = root / "environment-verified.json"
    marker.unlink(missing_ok=True)
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    # lambda: Arrow's jemalloc background thread crashes after DS/TorchCodec imports.
    os.environ["JE_ARROW_MALLOC_CONF"] = "background_thread:false"
    with patch("requests.sessions.Session.request", side_effect=RuntimeError("Unexpected HTTP")):
        for name in ("deepspeed", "flash_attn_2_cuda", "torchcodec"):
            print(f"Importing {name}", flush=True)
            import_module(name)
        print("Importing GR00T data and processor modules", flush=True)
        from gr00t.configs.data.chip_native_config import chip_native_modality_config
        from gr00t.data.dataset.chip_native_dataset import ChipNativeDataset
        from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import build_processor
        import numpy as np
        from PIL import Image
        from safetensors import safe_open
        import torch
        import torchvision

        assert torch.version.cuda == "12.8", torch.version.cuda
        print("Checking TorchVision and reading dataset", flush=True)
        torchvision.ops.nms(torch.tensor([[0.0, 0.0, 1.0, 1.0]]), torch.tensor([1.0]), 0.5)
        dataset = ChipNativeDataset(os.environ["CHIP_DATASET_ROOT"], chip_native_modality_config())
        step = dataset.get_step(0)
        assert step.actions["native_relative"].shape == (24, 37)
        assert np.isfinite(step.actions["native_relative"]).all()
        print("Loading offline Cosmos processor", flush=True)
        processor = build_processor("nvidia/Cosmos-Reason2-2B", {"local_files_only": True})
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.fromarray(image)}
                    for image in step.images["camera0_rgb"]
                ]
                + [{"type": "text", "text": "Perform the demonstrated task."}],
            }
        ]
        batch = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        assert batch["input_ids"].numel() > 0 and batch["pixel_values"].numel() > 0
        base = Path(os.environ["BASE_MODEL_PATH"])
        index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
        for filename in sorted(set(index.values())):
            with safe_open(base / filename, framework="pt", device="cpu") as weights:
                for key in (k for k, file in index.items() if file == filename):
                    weights.get_slice(key).get_shape()
        print(
            "Offline processor, dataset, weight index and compiled library imports passed",
            flush=True,
        )

    # Compile for the actual A100 architecture without requiring a GPU.
    with tempfile.TemporaryDirectory(dir=root / "tmp") as temporary:
        source = Path(temporary) / "check.cu"
        source.write_text("__global__ void check(float *x) { x[threadIdx.x] += 1.0f; }\n")
        subprocess.run(
            ["nvcc", "-arch=sm_80", "-c", str(source), "-o", str(source.with_suffix(".o"))],
            check=True,
        )
    result = {
        "passed": True,
        "gpu_tested": False,
        "runtime_environment": {"JE_ARROW_MALLOC_CONF": os.environ["JE_ARROW_MALLOC_CONF"]},
        "project": str(project),
        "revision": os.environ["CHIP_BUNDLE_REVISION"],
        "deployment_env_sha256": hashlib.sha256((root / "deployment.env").read_bytes()).hexdigest(),
        "versions": {
            p: version(p)
            for p in (
                "torch",
                "torchvision",
                "transformers",
                "flash-attn",
                "deepspeed",
                "torchcodec",
            )
        },
        "train_episodes": len(dataset.episode_names),
        "train_samples": len(dataset.samples),
        "cuda_compile_arch": "sm_80",
    }
    marker.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
