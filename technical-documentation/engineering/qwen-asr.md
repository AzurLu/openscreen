# Local Qwen transcription on Apple Silicon

Qwen is an optional transcription provider. The default remains the bundled
Whisper helper. With `provider: "qwen3-mlx"`, OpenScreen starts a local Python
sidecar using Qwen3-ASR-1.7B and Qwen3-ForcedAligner-0.6B, both MLX 8-bit.
ASR produces text; the aligner produces real word/character timestamps for
captions, highlighting, and transcript-based editing. Recordings stay local.

## Requirements and installation

Use macOS on Apple Silicon and a native arm64 Python 3.10 or newer; Python 3.11
is recommended. Intel Macs, Rosetta Python, Windows, and Linux should keep
using Whisper. Model weights total about 3.74 GB on disk; allow additional
space for the Python environment and download cache. Runtime memory depends
on audio length; both models remain loaded while the helper is running.

From the repository root:

```sh
python3.11 tools/qwen-asr/setup_runtime.py
```

This creates `~/Library/Application Support/openscreen/qwen3-runtime`, installs
the pinned requirements in `.venv`, downloads the pinned public Hugging Face
snapshots, verifies their files, and copies `qwen_server.py` into the runtime.
It prints a configuration but does **not** enable the provider. Choose another
location with `--runtime-dir /absolute/path/to/qwen3-runtime`.

Enable it by explicitly choosing the application's `stt-provider.json` path:

```sh
python3.11 tools/qwen-asr/setup_runtime.py --reuse-runtime \
  --write-config "$HOME/Library/Application Support/openscreen/stt-provider.json"
```

The file belongs in Electron's `app.getPath("userData")` directory. Development
builds can use a different directory from an installed app; pass that build's
path when configuring it. The application reads the file when starting, so
restart OpenScreen after changing it. All configured paths must be absolute.

`--reuse-runtime` verifies installed package versions and all model hashes
offline, then refreshes the sidecar. It does not install packages or download
weights. It works with an existing runtime using the same `.venv` and
`models/<model-name>` layout. An existing application configuration is never
overwritten unless `--overwrite-config` is also present. To switch back, set
the configuration to `{"provider":"whisper"}` or remove it and restart.

The generated structure is shown in
[`stt-provider.example.json`](../../tools/qwen-asr/stt-provider.example.json).
The setup copies the server into the runtime, so a packaged app does not need
the source checkout to remain in place. Model weights and Python packages are
external prerequisites; they are not bundled into OpenScreen releases.

## Pinned dependencies and checkpoints

[`requirements.txt`](../../tools/qwen-asr/requirements.txt) pins
`mlx-audio[stt]==0.6.5` and `opencc-python-reimplemented==0.1.7`.
`opencc` is the Python import name; it is not the package to install.

| Checkpoint | Immutable revision | Weights SHA-256 |
|---|---|---|
| [Qwen3-ASR-1.7B-8bit](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit) | `a8379a2e2f9e313c9292cdf1af4055ab56d50d55` | `bf304b009cc7eca79283056f787b44c952d24ac22cec787b39732bba3c23c13c` |
| [Qwen3-ForcedAligner-0.6B-8bit](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-8bit) | `0e1a68e91d815300c7c9754b2a7639378b23db15` | `be19ef8ac4326d032e7673342930b14c2df30bd68c1632493b0f563e30829f91` |

[`verify_models.py`](../../tools/qwen-asr/verify_models.py) uses the standard
Hugging Face downloader with these revisions. It checks weight sizes and
SHA-256, plus Git blob hashes for tokenizer/configuration files. To inspect
the manifest or verify an existing installation without network access:

```sh
python3 tools/qwen-asr/verify_models.py --list
python3 tools/qwen-asr/verify_models.py \
  --models-dir "$HOME/Library/Application Support/openscreen/qwen3-runtime/models"
```

## Inference and protocol

The Electron main process launches the configured Python executable and
server with `--host 127.0.0.1 --port <port> --asr-model <directory>
--aligner-model <directory>`. The sidecar only accepts pre-downloaded local
directories. There are no model downloads during transcription.

- `GET /` (or `/health`) returns `503` while loading and `200` with
  `{"ready":true,"backend":"qwen3-mlx",...}` only after both models load.
- `POST /inference` accepts a multipart `file` containing mono 16 kHz 16-bit
  PCM WAV, `response_format=verbose_json`, `language`, and optional `prompt`.
  Request limits are 64 MiB and 300 seconds. OpenScreen sends 30-second chunks;
  the sidecar also bounds longer requests to 30 seconds at quiet frames.
- The response contains `segments[].text`, `start`, `end`, and
  `words[]: {word,start,end,anchor}`. Times are seconds relative to the uploaded
  WAV; the application adds its outer chunk offset. `detected_language`,
  `timing`, and optional `peak_mlx_bytes` provide diagnostics.
- `POST /cancel` invalidates the current and queued requests. A blocking MLX
  call finishes before cancellation takes effect at the next model boundary;
  its stale result is discarded with HTTP `409`.

GPU inference is serialized, and both models stay resident between requests.
The server binds only to loopback and rejects requests containing an `Origin`
header. Its logs contain timing/status diagnostics, not audio or transcripts.

`language=auto` uses the ASR model's detected language. Explicit supported
languages are Chinese (`zh`), Cantonese (`yue`), English, German, Spanish,
French, Italian, Portuguese, Russian, Korean, and Japanese. The timestamp
aligner's supported languages constrain the combined provider. Unsupported
or undetected languages return `422` instead of guessed timestamps. The
existing `Glossary: term, term.` prompt is converted into Qwen hotwords.

Chinese/Cantonese output passes through local OpenCC `t2s` **before** forced
alignment. Punctuation is reattached to the aligned words so concatenating
them reproduces the segment text. Unmatched alignment text, invalid times,
or truncated ASR output fails the request; timestamps and confidence values
are not fabricated. The near-zero PCM silence gate only handles digital
silence; it is not a general speech detector for background noise.

## Verification

The tests run without MLX or downloaded weights, including a loopback HTTP
test using a fake engine:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tools/qwen-asr/tests -p 'test_*.py'
```

`tests/test_live_adapter.py` optionally checks an already running real server.
It does not start a server or download any data. Provide your own public or
synthetic fixtures in a directory: `chinese.wav`, `expected-chinese.txt`,
`qwen-official-chinese.wav`, `long-chinese.wav` (over 30 seconds), and
`silence.wav`, all mono 16 kHz PCM16. The Chinese fixtures must contain
Mandarin. Run with the runtime's Python so OpenCC is available:

```sh
"$HOME/Library/Application Support/openscreen/qwen3-runtime/.venv/bin/python" \
  tools/qwen-asr/tests/test_live_adapter.py \
  --base-url http://127.0.0.1:18765 --fixtures-dir /path/to/fixtures \
  --output /path/to/report.json --test-cancel
```

It checks consecutive requests, Chinese character error rate, simplified
text, exact word/text coverage, timestamp bounds and long-input offsets,
silence, invalid WAV, Origin rejection, and optionally in-flight cancellation.
Its report includes fixture transcripts; use public/synthetic audio when
sharing that report. Native GPU, application, and export checks still require
the target Mac; the dependency-free tests do not establish model accuracy.
