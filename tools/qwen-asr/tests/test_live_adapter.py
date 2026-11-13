#!/usr/bin/env python3
"""Validate an already running local Qwen sidecar using public/synthetic audio.

This script never starts a server or downloads/loads model weights. Run it with
the runtime's Python so OpenCC is available for the independent script check.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import threading
import time
import unicodedata
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import uuid
import wave


def request(base_url, path, *, method="GET", data=None, headers=None, timeout=300):
    started = time.monotonic()
    req = Request(base_url + path, data=data, headers=headers or {}, method=method)
    try:
        response = urlopen(req, timeout=timeout)
    except HTTPError as exc:
        response = exc
    with response:
        status = response.status
        raw = response.read()
        content_type = response.headers.get("Content-Type", "")
    if "application/json" not in content_type:
        raise AssertionError(f"HTTP {status} was not JSON")
    value = json.loads(raw)
    return status, value, time.monotonic() - started


def multipart(audio, *, language="auto", prompt=""):
    boundary = "openscreen-qwen-live-" + uuid.uuid4().hex
    body = []
    for name, value in [("response_format", "verbose_json"), ("language", language), ("prompt", prompt)]:
        body.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    body.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="fixture.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode())
    body.append(audio)
    body.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(body), {"Content-Type": f"multipart/form-data; boundary={boundary}"}


def wav_duration(path):
    with wave.open(str(path), "rb") as stream:
        assert stream.getframerate() == 16000, "Fixture must be 16 kHz"
        assert stream.getnchannels() == 1, "Fixture must be mono"
        assert stream.getsampwidth() == 2, "Fixture must be 16-bit PCM"
        return stream.getnframes() / stream.getframerate()


def cer_text(text, converter):
    text = unicodedata.normalize("NFC", converter.convert(text)).lower()
    return "".join(char for char in text if unicodedata.category(char)[0] in ("L", "N"))


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def validate_transcription(result, duration, converter, *, expected=None, max_cer=.15, long_audio=False):
    assert result.get("backend") == "qwen3-mlx", "Unexpected backend"
    assert result.get("detected_language") == "zh", "Chinese fixture was not detected as zh"
    segments = result.get("segments")
    assert isinstance(segments, list) and segments, "No transcript returned"
    if long_audio:
        assert duration > 30, "Long fixture does not cross the 30-second boundary"
        assert len(segments) >= 2, "Long input was not split into bounded chunks"
    last_start = 0.0
    words_count = 0
    zero_count = 0
    last_end = 0.0
    texts = []
    for segment in segments:
        text = segment.get("text")
        assert isinstance(text, str) and text.strip(), "Empty segment text"
        assert converter.convert(text) == text, "Transcript contains traditional characters"
        words = segment.get("words")
        assert isinstance(words, list) and words, "Missing aligned words"
        assert "".join(word["word"] for word in words) == text, "Word tokens lost or changed transcript characters/punctuation"
        for word in words:
            assert isinstance(word.get("word"), str) and word["word"], "Invalid word text"
            start, end = word.get("start"), word.get("end")
            assert isinstance(start, (int, float)) and isinstance(end, (int, float)), "Non-numeric word times"
            assert math.isfinite(start) and math.isfinite(end), "Non-finite word times"
            assert 0 <= start <= end <= duration + .002, "Word timestamps are outside the uploaded audio"
            assert start >= last_start - .001, "Word start timestamps are out of order"
            if "anchor" in word:
                assert 0 <= word["anchor"] <= duration + .002, "Anchor outside audio"
            assert "probability" not in word, "Adapter must not fabricate word confidence"
            zero_count += end == start
            words_count += 1
            last_start = start
            last_end = max(last_end, end)
        assert abs(segment["start"] - words[0]["start"]) <= .001
        assert abs(segment["end"] - max(word["end"] for word in words)) <= .001
        texts.append(text)
    assert words_count > 0 and zero_count < words_count, "No positive-duration aligned word"
    if long_audio:
        assert last_end > 30, "Long transcript lost its second chunk offset"
    timing = result.get("timing", {})
    assert abs(timing.get("audio_s", -1) - duration) < .005, "Reported audio duration differs from input"
    assert timing.get("elapsed_s", -1) >= 0
    assert timing.get("rtf", -1) >= 0
    peak = result.get("peak_mlx_bytes")
    if peak is not None:
        assert isinstance(peak, int) and peak > 0
    transcript = "".join(texts)
    detail = {"audio_seconds": round(duration, 3), "segment_count": len(segments),
              "word_count": words_count, "zero_duration_words": zero_count,
              "last_word_end": last_end, "text": transcript,
              "timing": timing, "peak_mlx_bytes": peak}
    if expected is not None:
        reference = cer_text(expected, converter)
        actual = cer_text(transcript, converter)
        assert reference, "Reference transcript is empty"
        error = edit_distance(reference, actual)
        cer = error / len(reference)
        detail.update({"cer": cer, "character_edits": error,
                       "reference_characters": len(reference), "maximum_cer": max_cer})
        if cer > max_cer:
            raise AssertionError(f"TTS character error rate {cer:.2%} exceeds {max_cer:.2%}; actual={transcript}")
    return detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--fixtures-dir", type=Path, required=True)
    parser.add_argument("--output", "--report", type=Path)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--wait-ready", type=float, default=120)
    parser.add_argument("--max-cer", type=float, default=.15)
    parser.add_argument("--test-cancel", action="store_true")
    parser.add_argument("--cancel-after", type=float, default=.5)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    parsed = urlsplit(base)
    if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost")
            or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment):
        parser.error("--base-url must be an HTTP loopback endpoint without path or credentials")
    if not 0 <= args.max_cer <= 1:
        parser.error("--max-cer must be between 0 and 1")
    from opencc import OpenCC
    converter = OpenCC("t2s")
    fixtures = args.fixtures_dir.resolve()
    report = {"base_url": base, "fixtures_dir": str(fixtures),
              "tests": [], "passed": False, "real_model_tests": True}

    def test(name, function):
        started = time.monotonic()
        try:
            details = function()
            entry = {"name": name, "status": "passed", "details": details}
        except Exception as exc:
            entry = {"name": name, "status": "failed", "error": str(exc), "exception": type(exc).__name__}
        entry["wall_seconds"] = round(time.monotonic() - started, 3)
        report["tests"].append(entry)
        print(f"{entry['status'].upper()}: {name} ({entry['wall_seconds']:.2f}s)", file=sys.stderr, flush=True)
        return entry["status"] == "passed"

    def ready():
        deadline = time.monotonic() + args.wait_ready
        last = None
        while True:
            try:
                status, health, _ = request(base, "/", timeout=5)
                last = health
                if status == 200 and health.get("ready"):
                    assert health.get("backend") == "qwen3-mlx"
                    return health
                if health.get("status") == "failed":
                    raise AssertionError(f"Model loading failed: {health}")
            except AssertionError:
                raise
            except Exception as exc:
                last = type(exc).__name__
            if time.monotonic() >= deadline:
                raise AssertionError(f"Service did not become ready: {last}")
            time.sleep(.5)

    def transcribe(filename, *, expected=None, long_audio=False):
        path = fixtures / filename
        duration = wav_duration(path)
        body, headers = multipart(path.read_bytes(), language="auto")
        status, result, wall = request(base, "/inference", method="POST", data=body,
                                       headers=headers, timeout=args.timeout)
        assert status == 200, f"Inference HTTP {status}: {result}"
        detail = validate_transcription(result, duration, converter, expected=expected,
                                        max_cer=args.max_cer, long_audio=long_audio)
        detail["request_seconds"] = round(wall, 3)
        return detail

    def silence():
        path = fixtures / "silence.wav"
        body, headers = multipart(path.read_bytes())
        status, result, _ = request(base, "/inference", method="POST", data=body,
                                   headers=headers, timeout=args.timeout)
        assert status == 200, result
        assert result.get("segments") == [], "Silent audio produced transcript text"
        assert abs(result["timing"]["audio_s"] - wav_duration(path)) < .005
        return {"segments": result["segments"], "timing": result["timing"]}

    def invalid_wav():
        body, headers = multipart(b"this is not an audio file")
        status, result, _ = request(base, "/inference", method="POST", data=body,
                                   headers=headers, timeout=10)
        assert status == 400, f"Invalid WAV returned HTTP {status}"
        assert isinstance(result.get("error"), str)
        return {"http_status": status, "error": result["error"]}

    def origin():
        status, result, _ = request(base, "/", headers={"Origin": "https://example.com"}, timeout=5)
        assert status == 403, "Cross-origin health request accepted"
        body, headers = multipart(b"invalid")
        headers["Origin"] = "https://example.com"
        status, result, _ = request(base, "/inference", method="POST", data=body, headers=headers, timeout=5)
        assert status == 403, "Cross-origin inference request accepted"
        return {"http_status": status}

    def cancel():
        body, headers = multipart((fixtures / "long-chinese.wav").read_bytes())
        outcome = {}
        started = threading.Event()

        def run():
            started.set()
            try:
                outcome["response"] = request(base, "/inference", method="POST", data=body,
                                               headers=headers, timeout=args.timeout)
            except Exception as exc:
                outcome["error"] = repr(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        started.wait(5)
        time.sleep(args.cancel_after)
        already_done = not thread.is_alive()
        status, ack, _ = request(base, "/cancel", method="POST", data=b"", timeout=5)
        assert status == 200 and ack.get("cancelled") is True, ack
        thread.join(args.timeout + 5)
        assert not thread.is_alive(), "Cancelled request did not finish at a model boundary"
        assert "error" not in outcome, outcome
        status, result, elapsed = outcome["response"]
        if already_done:
            assert status == 200
            return {"cancellation_acknowledged": True, "request_finished_before_cancel": True,
                    "inflight_cancellation_verified": False, "request_seconds": elapsed}
        assert status == 409 and result.get("kind") == "cancelled", f"Stale inference survived cancel: HTTP {status}"
        return {"cancellation_acknowledged": True, "inflight_cancellation_verified": True,
                "request_http_status": status, "request_seconds": elapsed}

    try:
        if test("health_ready", ready):
            expected = (fixtures / "expected-chinese.txt").read_text().strip()
            test("chinese_tts_first_request", lambda: transcribe("chinese.wav", expected=expected))
            test("official_chinese_second_request", lambda: transcribe("qwen-official-chinese.wav"))
            test("long_chinese_chunk_offsets", lambda: transcribe("long-chinese.wav", long_audio=True))
            test("silent_audio", silence)
            test("invalid_wav", invalid_wav)
            test("browser_origin_blocked", origin)
            if args.test_cancel:
                test("cancel_inflight", cancel)
        report["passed"] = bool(report["tests"]) and all(t["status"] == "passed" for t in report["tests"])
    finally:
        encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded)
        print(encoded, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
