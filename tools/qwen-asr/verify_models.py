#!/usr/bin/env python3
"""Download or verify the two pinned, public MLX Qwen checkpoints.

Verification is offline and uses only the Python standard library. --download
additionally needs huggingface_hub; downloads are sequential and its local_dir
cache automatically resumes interrupted transfers. No model code is executed.
Only the two named model directories under --models-dir are written.

Examples:
  python verify_models.py --list
  python verify_models.py --models-dir ./models --download
  python verify_models.py --models-dir ./models

Metadata was read from the Hugging Face model API with blobs=true on 2026-11-13.
Each revision is immutable. LFS files are checked with SHA-256; ordinary files
are checked using Git's SHA-1 blob framing, not plain file SHA-1. A SHA-256
inventory is written inside each model directory only after full verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


# filename -> (bytes, Git blob SHA-1). Weight files use SHA-256 separately.
COMMON = {
    ".gitattributes": (1519, "a6344aac8c09253b3b630fb776ae94478aa0275b"),
    "chat_template.json": (1161, "c44736493efd71ec96218cc626904698cdb13235"),
    "merges.txt": (1671853, "31349551d90c7606f325fe0f11bbb8bd5fa0d7c7"),
    "preprocessor_config.json": (330, "8f7f07346466d5d494ec0d4969d1c3d0190eed72"),
    "vocab.json": (2776833, "4783fe10ac3adce15ac8f358ef5462739852c569"),
}

MODELS = (
    {
        "repo_id": "mlx-community/Qwen3-ASR-1.7B-8bit",
        "revision": "a8379a2e2f9e313c9292cdf1af4055ab56d50d55",
        "weight_bytes": 2463307541,
        "weight_sha256": "bf304b009cc7eca79283056f787b44c952d24ac22cec787b39732bba3c23c13c",
        "files": {
            **COMMON,
            "README.md": (1008, "b835cd3f6d4f5a2dd6b7f3ad83138166bdf5f1b5"),
            "config.json": (7188, "888cc7fae19ea9ede543b1ae82e9c78aa06104ca"),
            "generation_config.json": (142, "7382a4d347c0a865b76bb1b8277f66a5ac312854"),
            "model.safetensors.index.json": (78968, "f0191caac2794d5b6fba0ab3eeb70f8eff90319e"),
            "tokenizer_config.json": (12487, "b93109843922a40c6654c5449d3bf95372267c66"),
        },
    },
    {
        "repo_id": "mlx-community/Qwen3-ForcedAligner-0.6B-8bit",
        "revision": "0e1a68e91d815300c7c9754b2a7639378b23db15",
        "weight_bytes": 1271924386,
        "weight_sha256": "be19ef8ac4326d032e7673342930b14c2df30bd68c1632493b0f563e30829f91",
        "files": {
            **COMMON,
            "README.md": (1068, "23c0a2762f86d9a54775cf58057bcb87ddd329da"),
            "config.json": (6940, "6a2c723df27c38e6924a3cfc6629636b42b79311"),
            "generation_config.json": (115, "b0a540032cf7b6b5ba1438d04ab69b7e787d9b8e"),
            "model.safetensors.index.json": (79108, "a360f0d6b31173a7285457c7b99613ba2b0a0653"),
            "tokenizer_config.json": (12666, "3df92df83b22342e36a7b11cf9a70de2a2996bec"),
        },
    },
)


def expected_files(model: dict) -> dict:
    result = {
        name: {"bytes": size, "git_blob_sha1": blob}
        for name, (size, blob) in model["files"].items()
    }
    result["model.safetensors"] = {
        "bytes": model["weight_bytes"],
        "sha256": model["weight_sha256"],
    }
    return result


def verify_file(path: Path, expected: dict) -> dict:
    if path.is_symlink():
        raise ValueError(f"Refusing a symlink: {path}")
    if not path.is_file():
        raise ValueError(f"Missing model file: {path}")
    size = path.stat().st_size
    if size != expected["bytes"]:
        raise ValueError(f"Wrong size for {path.name}: {size}, expected {expected['bytes']}")
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            sha256.update(block)
            if "git_blob_sha1" in expected:
                git_blob.update(block)
    digest = sha256.hexdigest()
    if "sha256" in expected and digest != expected["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")
    if "git_blob_sha1" in expected and git_blob.hexdigest() != expected["git_blob_sha1"]:
        raise ValueError(f"Git blob SHA-1 mismatch: {path}")
    return {**expected, "sha256": digest}


def verify_model(model: dict, base: Path, download: bool, write_manifest: bool = False) -> None:
    target = base / model["repo_id"].split("/", 1)[1]
    if target.is_symlink():
        raise ValueError(f"Refusing a model directory symlink: {target}")
    expected = expected_files(model)
    if download:
        from huggingface_hub import hf_hub_download

        target.mkdir(parents=True, exist_ok=True)
    # Never leave a stale success marker when beginning a fresh verification.
    marker = target / ".openscreen-model-verification.json"
    if download or write_manifest:
        marker.unlink(missing_ok=True)
    verified = {}
    # Small configuration/tokenizer files first, large weights last.
    for name, file_metadata in expected.items():
        destination = target / name
        print(f"Checking {model['repo_id']}/{name}", flush=True)
        try:
            verified[name] = verify_file(destination, file_metadata)
            continue
        except ValueError:
            if not download:
                raise
            if destination.is_symlink():
                raise
        print(f"Downloading pinned {name}", flush=True)
        hf_hub_download(
            repo_id=model["repo_id"],
            filename=name,
            revision=model["revision"],
            local_dir=str(target),
            token=False,
            # An existing incorrect file must not be accepted from local cache.
            force_download=destination.exists(),
        )
        verified[name] = verify_file(destination, file_metadata)
    if download or write_manifest:
        report = {
            "repo_id": model["repo_id"],
            "revision": model["revision"],
            "source": f"https://huggingface.co/{model['repo_id']}/tree/{model['revision']}",
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "files": verified,
        }
        temporary = marker.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, marker)
    print(f"Verified {model['repo_id']} at {model['revision']}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, help="Parent directory containing the two model directories")
    parser.add_argument("--download", action="store_true", help="Download/resume missing files and replace invalid files; otherwise verify offline")
    parser.add_argument("--write-manifest", action="store_true", help="Write the SHA-256 inventory after successful offline verification")
    parser.add_argument("--list", action="store_true", help="Print pinned revisions, file sizes, and hashes without accessing the network or filesystem")
    args = parser.parse_args()
    if args.list:
        print(json.dumps([
            {"repo_id": model["repo_id"], "revision": model["revision"], "files": expected_files(model)}
            for model in MODELS
        ], indent=2))
        return 0
    if args.models_dir is None:
        parser.error("--models-dir is required unless using --list")
    try:
        for model in MODELS:
            verify_model(model, args.models_dir.expanduser().resolve(), args.download, args.write_manifest)
    except (OSError, ValueError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Both pinned Qwen model snapshots are verified.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
