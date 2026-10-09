"""Package the CHIP training bundle and optionally upload it to a private HF model repo."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile

from huggingface_hub import HfApi, snapshot_download


MODELS = {
    "gr00t": ("nvidia/GR00T-N1.7-3B", "2fc962b973bccdd5d8ce4f67cc63b264d6886495"),
    "cosmos": ("nvidia/Cosmos-Reason2-2B", "9ce19a195e423419c349abfc86fd07178b230561"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def archive_files(destination, root, paths):
    """Regular files only, deterministic headers, no hardlink dependencies."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with open(temporary, "wb") as out:
        with gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0, compresslevel=1) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                for path in sorted(paths):
                    resolved = path.resolve()
                    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
                        raise ValueError(f"Archive file outside source root or not regular: {path}")
                    info = tarfile.TarInfo(str(path.relative_to(root)))
                    info.size = path.stat().st_size
                    info.mode = 0o755 if os.access(path, os.X_OK) else 0o644
                    info.uid = info.gid = info.mtime = 0
                    with open(path, "rb") as f:
                        archive.addfile(info, f)
    temporary.replace(destination)


def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise ValueError(f"Existing staged file differs: {destination}")
        return
    try:
        os.link(source.resolve(), destination)
    except OSError:
        shutil.copy2(source, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="mimiclite-chip/GR00T-CHIP")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    stage = args.staging_dir.resolve()
    data = args.dataset_path.resolve()
    if stage.is_relative_to(data) or data.is_relative_to(stage):
        raise ValueError("Staging and source dataset must be separate directories")
    stage.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    manifest = {"schema_version": 1, "repo_id": args.repo_id, "models": {}, "files": {}}

    for name, (upstream, revision) in MODELS.items():
        print(f"Preparing {upstream}@{revision}", flush=True)
        source = Path(snapshot_download(upstream, revision=revision))
        destination = stage / "models" / name
        for path in source.rglob("*"):
            if path.is_file() and ".cache" not in path.relative_to(source).parts:
                link_or_copy(path, destination / path.relative_to(source))
        manifest["models"][name] = {
            "upstream": upstream,
            "revision": revision,
            "directory": f"models/{name}",
        }
        if name == "cosmos":
            license_text = api.model_info(upstream, revision=revision).card_data.to_dict()[
                "extra_gated_prompt"
            ]
            (destination / "LICENSE").write_text(license_text + "\n")
            (destination / "NOTICE").write_text(
                "Licensed by NVIDIA Corporation under the NVIDIA Open Model License\n"
                "Built on NVIDIA Cosmos\n"
            )

    data_manifest = json.loads((data / "dataset_manifest.json").read_text())
    episodes = data_manifest["episode_names"]
    if len(episodes) != len(set(episodes)):
        raise ValueError("Duplicate episode names")
    actual_dirs = {p.name for p in data.iterdir() if p.is_dir()}
    if actual_dirs != set(episodes):
        raise ValueError("Dataset directories differ from manifest episode names")
    root_files = []
    for path in data.iterdir():
        if path.is_file():
            target = stage / "data/chip" / path.name
            link_or_copy(path, target)
            root_files.append(str(target.relative_to(stage)))
    archives = []
    for index, name in enumerate(episodes):
        episode = data / name
        if not episode.resolve().is_relative_to(data) or Path(name).name != name:
            raise ValueError(f"Invalid episode path: {name}")
        target = stage / "data/chip/episodes" / f"{index:05d}.tar.gz"
        archive_files(target, data, [p for p in episode.rglob("*") if p.is_file()])
        archives.append({"episode": name, "path": str(target.relative_to(stage))})
        if index % 25 == 0:
            print(f"Packed {index + 1}/{len(episodes)} episodes", flush=True)
    manifest["dataset"] = {
        "root_files": root_files,
        "archives": archives,
        "episodes": len(episodes),
        "source_frames": data_manifest["source_frames"],
    }

    code_files = [
        repo / p for p in ("pyproject.toml", "uv.lock", "LICENSE", "ATTRIBUTIONS.md", "README.md")
    ]
    for folder in (repo / "gr00t", repo / "examples/CHIP"):
        code_files.extend(
            p
            for p in folder.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"}
        )
    manifest["code_archive"] = "code/isaac-gr00t-chip.tar.gz"
    archive_files(stage / manifest["code_archive"], repo, code_files)
    for name in ("deploy.sh", "restore_hf_bundle.py"):
        shutil.copy2(repo / "examples/CHIP" / name, stage / name)
    (stage / "README.md").write_text(
        "---\nlicense: other\nlicense_name: nvidia-gr00t-and-cosmos-licenses\n"
        f"license_link: https://huggingface.co/{args.repo_id}/blob/main/models/gr00t/LICENSE\n"
        "tags:\n- robotics\n- gr00t\n---\n\n"
        "# GR00T CHIP training bundle\n\n"
        "Private backup of the CHIP dataset, adapted Isaac-GR00T source and complete upstream "
        "GR00T N1.7 / Cosmos-Reason2-2B snapshots. Built on NVIDIA Cosmos.\n\n"
        "GR00T weights are restricted to non-commercial research/evaluation by their LICENSE. "
        "Cosmos has its own NVIDIA Open Model License. Original model cards, licenses and "
        "notices are retained under models/. Dataset access is private; no public dataset license "
        "is granted by this backup.\n\n"
        "On a Linux x86_64 GPU server with a CUDA 12.8-compatible driver, CUDA toolkit "
        "(nvcc) and FFmpeg 4–7 runtime libraries, log in to an HF account with access to this repo. "
        "Reserve approximately 100 GB for dependencies, assets and training checkpoints.\n\n"
        "```bash\n"
        f"hf download {args.repo_id} deploy.sh --local-dir ./gr00t-bootstrap && "
        "bash ./gr00t-bootstrap/deploy.sh\n```\n\n"
        "Default: prepare environment/assets and run four-GPU training (10,000 steps, "
        "batch 32 after accumulation, 24 action rows, RTC delay 6). "
        "Set CUDA_VISIBLE_DEVICES to your allocated four GPUs; inherited scheduler settings "
        "are preserved. Set DEPLOY_ROOT to a large persistent disk.\n\n"
        "```bash\n"
        "DEPLOY_MODE=smoke CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./gr00t-bootstrap/deploy.sh\n"
        "DEPLOY_MODE=prepare bash ./gr00t-bootstrap/deploy.sh\n```\n\n"
        "smoke runs 10 optimizer steps and saves a checkpoint; prepare only installs/downloads. "
        "Extra CLI arguments are forwarded to the CHIP fine-tuning launcher. Assets are SHA-256 "
        "verified and model loading runs offline after download, using copies from this repo. "
        "The source archive contains examples/CHIP/README.md with training details.\n"
    )
    for path in sorted(stage.rglob("*")):
        rel = path.relative_to(stage)
        if path.is_file() and ".cache" not in rel.parts and path.name != "bundle.json":
            manifest["files"][str(rel)] = {"sha256": sha256(path), "size": path.stat().st_size}
            if path.stat().st_size > 100_000_000:
                print(f"Hashed {rel}: {path.stat().st_size} bytes", flush=True)
    manifest["bundle_id"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()[:20]
    (stage / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(manifest['files'])} files, bundle {manifest['bundle_id']}", flush=True)
    if args.upload:
        api.create_repo(args.repo_id, repo_type="model", private=True, exist_ok=True)
        if not api.repo_info(args.repo_id, repo_type="model").private:
            raise ValueError("Refusing to upload the raw dataset to a public repo")
        api.upload_large_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=stage,
            num_workers=4,
            print_report_every=30,
        )
        print("Uploaded revision:", api.repo_info(args.repo_id).sha, flush=True)


if __name__ == "__main__":
    main()
