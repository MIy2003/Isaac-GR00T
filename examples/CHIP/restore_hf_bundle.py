"""Download, verify and restore a pinned private HF training bundle."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import tarfile

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_path(root, relative):
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"Invalid bundle path: {relative}")
    return path


def extract_regular_files(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            safe_path(destination, member.name)
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Archive links/devices are not allowed: {member.name}")
            tar.extract(member, destination, filter="data")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    revision = HfApi().repo_info(args.repo_id, revision=args.revision).sha
    manifest_file = hf_hub_download(args.repo_id, "bundle.json", revision=revision)
    manifest = json.loads(Path(manifest_file).read_text())
    if manifest["schema_version"] != 1 or manifest["repo_id"] != args.repo_id:
        raise ValueError("Unexpected bundle schema or repository")
    bundle_id = manifest["bundle_id"]
    if not re.fullmatch(r"[0-9a-f]{20}", bundle_id):
        raise ValueError("Invalid bundle ID")
    payload = {k: v for k, v in manifest.items() if k != "bundle_id"}
    if hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20] != bundle_id:
        raise ValueError("Bundle manifest digest mismatch")
    for name in manifest["files"]:
        safe_path(root, name)
    snapshot = Path(
        snapshot_download(
            args.repo_id, revision=revision, allow_patterns=list(manifest["files"]), max_workers=4
        )
    )
    for index, (name, expected) in enumerate(manifest["files"].items()):
        path = safe_path(snapshot, name)
        if path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Bundle checksum mismatch: {name}")
        if index % 25 == 0 or expected["size"] > 100_000_000:
            print(f"Verified {index + 1}/{len(manifest['files'])}: {name}", flush=True)

    release = root / "releases" / bundle_id
    project = release / "Isaac-GR00T"
    dataset = release / "data/chip"
    marker = release / "RESTORED.json"
    if not marker.exists():
        # Extract in a fresh directory; incomplete attempts cannot supply symlinks
        # or half-written files on a retry. Never overwrite a user's existing tree.
        temporary = root / "releases" / f".{bundle_id}-{os.getpid()}"
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            extract_regular_files(snapshot / manifest["code_archive"], temporary / "Isaac-GR00T")
            for source in manifest["dataset"]["root_files"]:
                target = temporary / "data/chip" / Path(source).name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(safe_path(snapshot, source), target)
            for item in manifest["dataset"]["archives"]:
                extract_regular_files(safe_path(snapshot, item["path"]), temporary / "data/chip")
            (temporary / "RESTORED.json").write_text(
                json.dumps({"revision": revision, "bundle_id": bundle_id})
            )
            temporary.rename(release)
        except BaseException:
            shutil.rmtree(temporary)
            raise

    # GR00T's checkpoint refers to the upstream Cosmos ID. Seed an isolated,
    # per-bundle HF cache with the exact licensed upstream bytes from our mirror.
    # This preserves checkpoint/config compatibility and needs no upstream token
    # or HTTP requests once the mirror has been downloaded.
    runtime_cache = release / "hf-runtime/hub"
    for model in manifest["models"].values():
        source = safe_path(snapshot, model["directory"])
        if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
            raise ValueError("Invalid upstream revision")
        cache = runtime_cache / ("models--" + model["upstream"].replace("/", "--"))
        target = cache / "snapshots" / model["revision"]
        target.mkdir(parents=True, exist_ok=True)
        for path in source.rglob("*"):
            if path.is_file():
                link = target / path.relative_to(source)
                link.parent.mkdir(parents=True, exist_ok=True)
                if not link.exists():
                    link.symlink_to(path.resolve())
        (cache / "refs").mkdir(exist_ok=True)
        (cache / "refs/main").write_text(model["revision"])
    variables = {
        "CHIP_PROJECT_ROOT": str(project),
        "CHIP_DATASET_ROOT": str(dataset),
        "BASE_MODEL_PATH": str(snapshot / manifest["models"]["gr00t"]["directory"]),
        "HF_HUB_CACHE": str(runtime_cache),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CHIP_BUNDLE_REVISION": revision,
    }
    (root / "deployment.env").write_text(
        "".join(f"export {key}={shlex.quote(value)}\n" for key, value in variables.items())
    )
    print(f"Restored bundle {bundle_id}; environment: {root / 'deployment.env'}", flush=True)


if __name__ == "__main__":
    main()
