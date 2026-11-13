# Local Qwen ASR for Apple Silicon

Optional Qwen3-ASR-1.7B and Qwen3-ForcedAligner-0.6B, both MLX 8-bit, behind the
same loopback transcription interface used by OpenScreen. Whisper remains the
default. See [setup and protocol](../../technical-documentation/engineering/qwen-asr.md).

```sh
python3.11 tools/qwen-asr/setup_runtime.py
python3 -m unittest discover -s tools/qwen-asr/tests -p 'test_*.py'
```

Setup prints the configuration without enabling it. `--write-config` enables
the provider; replacing an existing configuration also requires
`--overwrite-config`. `--reuse-runtime` verifies existing dependencies and
weights offline and updates the sidecar without downloading them again.

No recordings, model weights, virtual environments, or generated reports are
included in this directory.
