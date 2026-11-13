import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


def load_module(name):
    path = Path(__file__).resolve().parents[1] / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup = load_module("setup_runtime")
verify = load_module("verify_models")


class SetupTests(unittest.TestCase):
    def test_config_keeps_virtualenv_interpreter_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            python = runtime / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.symlink_to(sys.executable)
            config = setup.provider_config(runtime)
            self.assertEqual(config["provider"], "qwen3-mlx")
            self.assertEqual(config["pythonPath"], str(runtime.resolve() / ".venv/bin/python"))
            self.assertNotEqual(config["pythonPath"], str(python.resolve()))
            self.assertTrue(all(Path(value).is_absolute() for key, value in config.items() if key != "provider"))

    def test_existing_config_needs_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stt-provider.json"
            original = '{"provider":"whisper"}\n'
            path.write_text(original)
            with self.assertRaises(FileExistsError):
                setup.write_config(path, {"provider": "qwen3-mlx"})
            self.assertEqual(path.read_text(), original)
            setup.write_config(path, {"provider": "qwen3-mlx"}, overwrite=True)
            self.assertEqual(json.loads(path.read_text()), {"provider": "qwen3-mlx"})
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_hash_verification_rejects_same_length_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.safetensors"
            payload = b"test model bytes"
            path.write_bytes(payload)
            metadata = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            self.assertEqual(verify.verify_file(path, metadata)["sha256"], metadata["sha256"])
            path.write_bytes(payload.upper())
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify.verify_file(path, metadata)

    def test_small_files_use_git_blob_hash_not_plain_sha1(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            payload = b'{"model":"qwen"}\n'
            path.write_bytes(payload)
            git_hash = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
            verify.verify_file(path, {"bytes": len(payload), "git_blob_sha1": git_hash})
            with self.assertRaisesRegex(ValueError, "Git blob SHA-1 mismatch"):
                verify.verify_file(path, {"bytes": len(payload), "git_blob_sha1": hashlib.sha1(payload).hexdigest()})

    def test_reuse_is_offline_and_does_not_create_application_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            commands = []
            with (patch.object(sys, "argv", ["setup_runtime.py", "--reuse-runtime", "--runtime-dir", temporary]),
                  patch.object(setup.sys, "platform", "darwin"),
                  patch.object(setup.sys, "version_info", (3, 11)),
                  patch.object(setup.platform, "machine", return_value="arm64"),
                  patch.object(setup, "verify_runtime"),
                  patch.object(setup, "run", side_effect=commands.append),
                  contextlib.redirect_stdout(io.StringIO()) as output,
                  contextlib.redirect_stderr(io.StringIO())):
                self.assertEqual(setup.main(), 0)
            self.assertEqual(len(commands), 1)
            self.assertIn("verify_models.py", commands[0][1])
            self.assertNotIn("--download", commands[0])
            self.assertEqual(json.loads(output.getvalue())["serverPath"], str(runtime.resolve() / "qwen_server.py"))
            self.assertEqual(list(runtime.iterdir()), [runtime / "qwen_server.py"])

    def test_existing_config_fails_before_setup_mutations(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stt-provider.json"
            path.write_text('{"provider":"whisper"}')
            with (patch.object(sys, "argv", ["setup_runtime.py", "--write-config", str(path)]),
                  patch.object(setup.sys, "platform", "darwin"),
                  patch.object(setup.sys, "version_info", (3, 11)),
                  patch.object(setup.platform, "machine", return_value="arm64"),
                  patch.object(setup, "run") as run,
                  contextlib.redirect_stderr(io.StringIO()),
                  self.assertRaises(SystemExit) as caught):
                setup.main()
            self.assertEqual(caught.exception.code, 2)
            run.assert_not_called()
            self.assertEqual(json.loads(path.read_text()), {"provider": "whisper"})


if __name__ == "__main__":
    unittest.main()
