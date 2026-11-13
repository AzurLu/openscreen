#!/usr/bin/env python3
"""Loopback-only Qwen3 MLX transcription bridge for OpenScreen.

The service keeps ASR and aligner models resident. Word timestamps are local to
each uploaded WAV; OpenScreen is responsible for adding its outer chunk offset.
No transcript, glossary, or audio is written to logs or temporary files.
"""
from __future__ import annotations

import argparse
import array
import io
import json
import logging
import math
import re
import signal
import sys
import threading
import time
import unicodedata
import wave
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

LOG = logging.getLogger("openscreen-qwen")
SAMPLE_RATE = 16000
MAX_BODY = 64 * 1024 * 1024
MAX_AUDIO_SECONDS = 300
LANGUAGES = {
    "zh": "Chinese", "yue": "Cantonese", "en": "English", "de": "German",
    "es": "Spanish", "fr": "French", "it": "Italian", "pt": "Portuguese",
    "ru": "Russian", "ko": "Korean", "ja": "Japanese",
}
LANGUAGE_CODES = {name.lower(): code for code, name in LANGUAGES.items()}


class AdapterError(Exception):
    def __init__(self, message: str, status: int = 400, kind: str = "invalid-request"):
        super().__init__(message)
        self.status = status
        self.kind = kind


class Cancelled(AdapterError):
    def __init__(self):
        super().__init__("Transcription cancelled", 409, "cancelled")


@dataclass(frozen=True)
class Audio:
    pcm: bytes
    sample_rate: int = SAMPLE_RATE

    @property
    def frames(self) -> int:
        return len(self.pcm) // 2

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate

    def slice(self, start: int, end: int) -> "Audio":
        return Audio(self.pcm[start * 2:end * 2], self.sample_rate)


def decode_wav(data: bytes) -> Audio:
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getcomptype()) != (1, SAMPLE_RATE, 2, "NONE"):
                raise AdapterError("Expected mono 16 kHz 16-bit PCM WAV")
            frames = wav.getnframes()
            if frames <= 0:
                raise AdapterError("Audio has no samples")
            if frames > SAMPLE_RATE * MAX_AUDIO_SECONDS:
                raise AdapterError("Audio exceeds the 300-second request limit", 413)
            pcm = wav.readframes(frames)
            if len(pcm) != frames * 2:
                raise AdapterError("Truncated WAV audio")
    except AdapterError:
        raise
    except (wave.Error, EOFError, ValueError, OSError) as exc:
        raise AdapterError("Unreadable PCM WAV") from exc
    return Audio(pcm)


def pcm_samples(audio: Audio) -> array.array:
    values = array.array("h")
    values.frombytes(audio.pcm)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def is_near_digital_silence(audio: Audio) -> bool:
    """Only gate near-zero input; this intentionally is not a general VAD."""
    values = pcm_samples(audio)
    if not values:
        return True
    peak = max(abs(value) for value in values)
    if peak > 4:  # Avoid suppressing quiet but nonzero speech.
        return False
    return sum(value * value for value in values) / len(values) <= 4


def split_audio(audio: Audio, max_seconds: float = 30.0) -> list[tuple[Audio, float]]:
    """Bound memory and latency; choose a quiet 20 ms frame before each cut."""
    limit = int(max_seconds * audio.sample_rate)
    if audio.frames <= limit:
        return [(audio, 0.0)]
    values = pcm_samples(audio)
    window = int(0.02 * audio.sample_rate)
    search = int(2.0 * audio.sample_rate)
    chunks = []
    start = 0
    while start < len(values):
        end = min(len(values), start + limit)
        if end < len(values):
            lower = max(start + window, end - search)
            choices = range(lower, end - window + 1, window)
            candidate = min(choices, key=lambda i: (sum(x * x for x in values[i:i + window]), abs(end - i)), default=end)
            end = max(start + window, candidate)
        chunks.append((audio.slice(start, end), start / audio.sample_rate))
        start = end
    return chunks


def language_code(value: Any, *, allow_auto: bool = True) -> str:
    if value is None or str(value).strip().lower() in ("", "auto", "none", "unknown"):
        if allow_auto:
            return "auto"
        raise AdapterError("ASR did not return a supported language", 422, "language-unavailable")
    name = str(value).strip().lower()
    if name in ("zh-cn", "zh-tw", "zh-hans", "zh-hant", "mandarin"):
        name = "zh"
    code = LANGUAGE_CODES.get(name, name)
    if code not in LANGUAGES:
        raise AdapterError("The local aligner does not support this language", 422, "unsupported-language")
    return code


def detected_language(result: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    candidates = getattr(result, "language", None)
    if isinstance(candidates, list):
        candidates = list(candidates)
    else:
        candidates = [candidates]
    for segment in getattr(result, "segments", None) or []:
        candidates.append(segment.get("language"))
    for value in candidates:
        if not value:
            continue
        for part in str(value).split(","):
            try:
                return language_code(part, allow_auto=False)
            except AdapterError:
                continue
    raise AdapterError("ASR did not return a supported language", 422, "language-unavailable")


def glossary_hotwords(prompt: str) -> list[str]:
    if len(prompt) > 4096:
        raise AdapterError("Glossary exceeds 4096 characters")
    prompt = re.sub(r"^\s*Glossary\s*:\s*", "", prompt, flags=re.IGNORECASE).strip().rstrip(".")
    words = [word.strip() for word in re.split(r"[,，;；\n]", prompt) if word.strip()]
    return list(dict.fromkeys(words))[:128]


def kept_character(char: str) -> bool:
    return char == "'" or unicodedata.category(char)[0] in ("L", "N")


def attach_text_to_alignment(text: str, items: Any, duration: float) -> list[dict[str, Any]]:
    """Preserve original punctuation instead of joining punctuation-free items.

    Require exact token/text coverage. Inventing evenly spaced timestamps or
    silently dropping an unmatched token would make text-driven cuts unsafe.
    """
    text = unicodedata.normalize("NFC", text)
    positions = [i for i, char in enumerate(text) if kept_character(char)]
    expected = "".join(text[i] for i in positions)
    records = []
    consumed = 0
    for item in items:
        token = unicodedata.normalize("NFC", str(getattr(item, "text", "")))
        token = "".join(char for char in token if kept_character(char))
        if not token:
            continue
        if not expected.startswith(token, consumed):
            raise AdapterError("Aligner tokens did not match the transcript", 502, "alignment-mismatch")
        try:
            start = float(item.start_time)
            end = float(item.end_time)
        except (AttributeError, ValueError, TypeError) as exc:
            raise AdapterError("Aligner returned invalid word times", 502, "alignment-invalid") from exc
        if not math.isfinite(start) or not math.isfinite(end):
            raise AdapterError("Aligner returned non-finite word times", 502, "alignment-invalid")
        if start < -0.25 or end > duration + 0.25 or end < start - 0.001:
            raise AdapterError("Aligner word times fell outside the audio", 502, "alignment-invalid")
        raw_start = positions[consumed]
        consumed += len(token)
        raw_end = positions[consumed - 1] + 1
        records.append({"raw_start": raw_start, "raw_end": raw_end,
                        "start": max(0.0, min(duration, start)),
                        "end": max(0.0, min(duration, end))})
    if consumed != len(expected):
        raise AdapterError("Aligner omitted transcript tokens", 502, "alignment-mismatch")
    if not records:
        return []
    # Punctuation and spaces between tokens go to the preceding token. Leading
    # punctuation stays on the first token, and no visible character is lost.
    words = []
    last_start = 0.0
    for index, record in enumerate(records):
        left = 0 if index == 0 else record["raw_start"]
        right = records[index + 1]["raw_start"] if index + 1 < len(records) else len(text)
        start = max(last_start, record["start"])
        end = max(start, record["end"])
        words.append({"word": text[left:right], "start": round(start, 3),
                      "end": round(end, 3), "anchor": round(start, 3)})
        last_start = start
    return words


class MLXEngine:
    def __init__(self, asr_path: str, aligner_path: str):
        self.asr_path = asr_path
        self.aligner_path = aligner_path

    def load(self):
        import mlx.core as mx
        import numpy as np
        from mlx_audio.stt import load
        from opencc import OpenCC
        # Only local, pre-downloaded paths are accepted. A typo must not cause
        # the service to download an unrelated repository at inference time.
        for path in (self.asr_path, self.aligner_path):
            if not Path(path).is_dir():
                raise RuntimeError("Configured local model directory is missing")
        self.mx = mx
        self.np = np
        self.converter = OpenCC("t2s")
        self.asr = load(self.asr_path)
        self.aligner = load(self.aligner_path)

    def samples(self, audio: Audio):
        return self.np.frombuffer(audio.pcm, dtype="<i2").astype(self.np.float32) / 32768.0

    def transcribe(self, audio: Audio, language: str, hotwords: list[str]):
        return self.asr.generate(self.samples(audio), language=None if language == "auto" else LANGUAGES[language],
                                 hotwords=hotwords or None, temperature=0.0, batch_size=1,
                                 chunk_duration=30.0, max_tokens=2048, verbose=False)

    def simplify(self, text: str) -> str:
        return self.converter.convert(text)

    def align(self, audio: Audio, text: str, language: str):
        return self.aligner.generate(audio=self.samples(audio), text=text, language=LANGUAGES[language])

    def reset_peak(self):
        if hasattr(self.mx, "reset_peak_memory"):
            self.mx.reset_peak_memory()

    def peak_memory(self):
        return int(self.mx.get_peak_memory()) if hasattr(self.mx, "get_peak_memory") else None


class Adapter:
    def __init__(self, engine: Any):
        self.engine = engine
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.ready = threading.Event()
        self.status = "loading"
        self.failure = None
        self.epoch = 0

    def load(self):
        try:
            self.engine.load()
            with self.state_lock:
                self.status = "ready"
            LOG.info("Both local MLX models loaded")
        except Exception as exc:
            with self.state_lock:
                self.status = "failed"
                self.failure = type(exc).__name__
            LOG.error("Model loading failed (%s)", type(exc).__name__)
        finally:
            self.ready.set()

    def health(self):
        with self.state_lock:
            return {"status": self.status, "ready": self.status == "ready", "backend": "qwen3-mlx",
                    "asr_model": Path(getattr(self.engine, "asr_path", "Qwen3-ASR-1.7B-8bit")).name,
                    "aligner_model": Path(getattr(self.engine, "aligner_path", "Qwen3-ForcedAligner-0.6B-8bit")).name,
                    **({"error": self.failure} if self.failure else {})}

    def cancel(self):
        with self.state_lock:
            self.epoch += 1
            return self.epoch

    def check_cancel(self, epoch: int):
        with self.state_lock:
            if epoch != self.epoch:
                raise Cancelled()

    def process(self, audio: Audio, language: str = "auto", prompt: str = ""):
        code = language_code(language)
        hotwords = glossary_hotwords(prompt)
        with self.state_lock:
            epoch = self.epoch
        if not self.ready.wait(120) or self.status != "ready":
            raise AdapterError("Local models are not ready", 503, "not-ready")
        with self.lock:
            self.check_cancel(epoch)
            started = time.monotonic()
            if hasattr(self.engine, "reset_peak"):
                self.engine.reset_peak()
            segments = []
            detected = None
            for chunk, offset in split_audio(audio):
                self.check_cancel(epoch)
                if is_near_digital_silence(chunk):
                    continue
                result = self.engine.transcribe(chunk, code, hotwords)
                self.check_cancel(epoch)
                if getattr(result, "generation_tokens", 0) >= 2048:
                    raise AdapterError("ASR reached its output limit; retry with shorter audio", 502, "output-truncated")
                text = unicodedata.normalize("NFC", str(getattr(result, "text", ""))).strip()
                if not any(kept_character(char) for char in text):
                    continue
                current_language = detected_language(result, code)
                if current_language in ("zh", "yue"):
                    text = self.engine.simplify(text)
                aligned = self.engine.align(chunk, text, current_language)
                self.check_cancel(epoch)
                words = attach_text_to_alignment(text, aligned, chunk.duration)
                if not words:
                    continue
                for word in words:
                    for field in ("start", "end", "anchor"):
                        word[field] = round(word[field] + offset, 3)
                segments.append({"text": text, "start": words[0]["start"],
                                 "end": max(word["end"] for word in words), "words": words})
                detected = detected or current_language
            self.check_cancel(epoch)
            elapsed = time.monotonic() - started
            peak = self.engine.peak_memory() if hasattr(self.engine, "peak_memory") else None
            LOG.info("Inference finished: audio=%.2fs elapsed=%.2fs segments=%d peak_mlx_bytes=%s",
                     audio.duration, elapsed, len(segments), peak)
            return {"segments": segments, "detected_language": detected or code, "backend": "qwen3-mlx",
                    "timing": {"elapsed_s": elapsed, "audio_s": audio.duration,
                               "rtf": elapsed / audio.duration},
                    **({"peak_mlx_bytes": peak} if peak is not None else {})}


def parse_multipart(content_type: str, body: bytes) -> tuple[bytes, dict[str, str]]:
    if not content_type.lower().startswith("multipart/form-data"):
        raise AdapterError("Expected multipart/form-data", 415)
    if "\r" in content_type or "\n" in content_type:
        raise AdapterError("Invalid Content-Type")
    try:
        envelope = b"Content-Type: " + content_type.encode("ascii", "strict") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    except UnicodeEncodeError as exc:
        raise AdapterError("Invalid Content-Type") from exc
    message = BytesParser(policy=policy.default).parsebytes(envelope)
    if not message.is_multipart() or message.defects:
        raise AdapterError("Malformed multipart request")
    file_data = None
    fields = {}
    for index, part in enumerate(message.iter_parts()):
        if index > 12 or part.is_multipart() or part.defects:
            raise AdapterError("Malformed multipart fields")
        name = part.get_param("name", header="content-disposition")
        payload = part.get_payload(decode=True)
        if payload is None:
            raise AdapterError("Malformed multipart payload")
        if name == "file":
            if file_data is not None:
                raise AdapterError("Only one audio file is accepted")
            file_data = payload
        elif name in ("language", "prompt", "response_format"):
            if name in fields or len(payload) > 16384:
                raise AdapterError("Invalid or duplicate multipart field")
            try:
                fields[name] = payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise AdapterError("Text fields must be UTF-8") from exc
    if not file_data:
        raise AdapterError("Missing audio file")
    if fields.get("response_format", "verbose_json") != "verbose_json":
        raise AdapterError("Only verbose_json output is supported")
    return file_data, fields


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = "OpenScreenQwen/1"

    def log_message(self, format, *args):
        # HTTP paths and exception text can contain user content; log summaries
        # explicitly above instead of the base handler's untrusted request line.
        pass

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def respond(self, status: int, value: dict):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def origin_allowed(self):
        # The Electron main process sends no Origin. Block browser-originated
        # requests so a website cannot consume GPU or cancel the user's job.
        return self.headers.get("Origin") is None

    def do_GET(self):
        if not self.origin_allowed():
            return self.respond(403, {"error": "Browser-originated requests are not accepted"})
        if self.path not in ("/", "/health"):
            return self.respond(404, {"error": "Not found"})
        health = self.server.adapter.health()
        self.respond(200 if health["ready"] else 503, health)

    def do_POST(self):
        try:
            if not self.origin_allowed():
                raise AdapterError("Browser-originated requests are not accepted", 403)
            if self.path not in ("/inference", "/cancel"):
                raise AdapterError("Not found", 404)
            if self.headers.get("Transfer-Encoding"):
                raise AdapterError("Chunked request bodies are not supported", 411)
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) > 1:
                raise AdapterError("Multiple Content-Length headers")
            try:
                length = int(lengths[0]) if lengths else 0
            except ValueError as exc:
                raise AdapterError("Invalid Content-Length") from exc
            if length < 0 or length > MAX_BODY:
                raise AdapterError("Request body too large", 413)
            if self.path == "/cancel":
                if length > 1024:
                    raise AdapterError("Cancellation body too large", 413)
                if length:
                    self.rfile.read(length)
                return self.respond(200, {"cancelled": True, "generation": self.server.adapter.cancel()})
            if not length:
                raise AdapterError("Content-Length is required", 411)
            body = self.rfile.read(length)
            if len(body) != length:
                raise AdapterError("Truncated request body")
            file_data, fields = parse_multipart(self.headers.get("Content-Type", ""), body)
            audio = decode_wav(file_data)
            result = self.server.adapter.process(audio, fields.get("language", "auto"), fields.get("prompt", ""))
            self.respond(200, result)
        except AdapterError as exc:
            self.respond(exc.status, {"error": str(exc), "kind": exc.kind})
        except (TimeoutError, ConnectionResetError, BrokenPipeError):
            self.respond(408, {"error": "Request connection timed out", "kind": "request-timeout"})
        except Exception as exc:
            LOG.error("Inference request failed (%s)", type(exc).__name__)
            self.respond(500, {"error": "Local transcription failed", "kind": "inference-failed"})


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False
    request_queue_size = 4

    def __init__(self, address, adapter):
        self.adapter = adapter
        super().__init__(address, Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1"])
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--asr-model", "--model", dest="asr_model", required=True)
    parser.add_argument("--aligner-model", required=True)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("Invalid port")
    logging.basicConfig(level=logging.INFO, format="[openscreen-qwen] %(message)s", stream=sys.stderr)
    engine = MLXEngine(str(Path(args.asr_model).expanduser().resolve()),
                       str(Path(args.aligner_model).expanduser().resolve()))
    adapter = Adapter(engine)
    server = Server((args.host, args.port), adapter)
    threading.Thread(target=adapter.load, daemon=True, name="qwen-model-loader").start()
    LOG.info("Loopback server listening on port %d", server.server_port)

    def stop(_signum, _frame):
        adapter.cancel()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
