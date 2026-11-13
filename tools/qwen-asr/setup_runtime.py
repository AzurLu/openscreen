#!/usr/bin/env python3
"""Set up the optional local Qwen runtime on Apple Silicon.

The default command installs the pinned dependencies and verified model snapshots,
then prints a provider configuration. It never changes OpenScreen's configuration
unless --write-config is supplied. --reuse-runtime skips installation/downloads;
it verifies an existing .venv and models directory before updating the sidecar.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
PINNED_PACKAGES = {"mlx-audio": "0.6.5", "opencc-python-reimplemented": "0.1.7"}


def provider_config(runtime: Path) -> dict[str, str]:
    # Keep the venv executable path: resolving its symlink selects the base
    # interpreter and loses the virtual environment's installed packages.
    runtime = runtime.expanduser().resolve()
    models = runtime / "models"
    return {
        "provider": "qwen3-mlx",
        "pythonPath": str(runtime / ".venv/bin/python"),
        "serverPath": str(runtime / "qwen_server.py"),
        "asrModelPath": str(models / "Qwen3-ASR-1.7B-8bit"),
        "alignerModelPath": str(models / "Qwen3-ForcedAligner-0.6B-8bit"),
    }


def write_config(path: Path, config: dict, *, overwrite: bool = False) -> None:
    encoded = json.dumps(config, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        # Exclusive creation also protects against another process creating a
        # configuration between the preflight check and this final write.
        with path.open("x", encoding="utf-8") as destination:
            destination.write(encoded)
        return
    replace_file(path, encoded)


def replace_file(path: Path, text: str) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as output:
            temporary = Path(output.name)
            output.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(command: list[str]) -> None:
    # Keep stdout as a machine-readable configuration; pip/download progress is
    # visible on stderr. Setup does not load models or access recordings.
    subprocess.run(command, check=True, stdout=sys.stderr,
                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


def verify_runtime(python: Path) -> None:
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("Runtime .venv/bin/python is missing; omit --reuse-runtime to install it")
    script = (
        "import platform, sys; from importlib.metadata import version; "
        "assert sys.version_info >= (3, 10), 'Python 3.10 or newer is required'; "
        "assert sys.platform == 'darwin' and platform.machine() == 'arm64', "
        "'The runtime must use native Apple Silicon Python'; "
        f"expected = {PINNED_PACKAGES!r}; "
        "actual = {name: version(name) for name in expected}; "
        "assert actual == expected, f'Runtime versions differ: {actual}'; "
        "import mlx.core; from mlx_audio.stt import load; from opencc import OpenCC"
    )
    run([str(python), "-c", script])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path,
                        default=Path.home() / "Library/Application Support/openscreen/qwen3-runtime")
    parser.add_argument("--reuse-runtime", action="store_true",
                        help="Use existing pinned dependencies/models offline; refresh only the sidecar")
    parser.add_argument("--write-config", type=Path,
                        help="Explicitly write this application's stt-provider.json after successful setup")
    parser.add_argument("--overwrite-config", action="store_true",
                        help="Allow replacing an existing --write-config file")
    args = parser.parse_args()
    if args.overwrite_config and args.write_config is None:
        parser.error("--overwrite-config requires --write-config")
    if sys.platform != "darwin" or platform.machine() != "arm64":
        parser.error("Qwen MLX requires macOS with native Apple Silicon Python")
    if sys.version_info < (3, 10):
        parser.error("Use Python 3.10 or newer; Python 3.11 is recommended")
    runtime = args.runtime_dir.expanduser().resolve()
    config_path = args.write_config.expanduser().absolute() if args.write_config else None
    if config_path and config_path.exists() and not args.overwrite_config:
        parser.error("Configuration already exists; use --overwrite-config to replace it explicitly")
    try:
        config = provider_config(runtime)
        python = Path(config["pythonPath"])
        if not args.reuse_runtime:
            runtime.mkdir(parents=True, exist_ok=True)
            if not python.exists():
                run([sys.executable, "-m", "venv", str(runtime / ".venv")])
            run([str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])
        verify_runtime(python)
        verification = [str(python), str(ROOT / "verify_models.py"),
                        "--models-dir", str(runtime / "models")]
        if not args.reuse_runtime:
            verification.append("--download")
        run(verification)
        replace_file(Path(config["serverPath"]), (ROOT / "qwen_server.py").read_text(encoding="utf-8"))
        if config_path:
            write_config(config_path, config, overwrite=args.overwrite_config)
            print(f"Provider configured at {config_path}; restart OpenScreen to use it.", file=sys.stderr)
        else:
            print("Runtime ready. App configuration was not changed; use --write-config to enable it.",
                  file=sys.stderr)
        print(json.dumps(config, indent=2))
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
