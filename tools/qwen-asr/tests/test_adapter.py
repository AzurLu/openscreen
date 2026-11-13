import http.client
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import threading
from types import SimpleNamespace
import unittest
import wave

path = Path(__file__).resolve().parents[1] / "qwen_server.py"
spec = importlib.util.spec_from_file_location("qwen_server", path)
q = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = q
spec.loader.exec_module(q)


def wav_bytes(values, rate=16000, channels=1):
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(struct.pack("<" + "h" * len(values), *values))
    return out.getvalue()


def item(text, start, end):
    return SimpleNamespace(text=text, start_time=start, end_time=end)


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.after_asr = None

    def load(self):
        self.calls.append("load")

    def transcribe(self, audio, language, hotwords):
        self.calls.append(("asr", language, hotwords))
        if self.after_asr:
            self.after_asr()
        return SimpleNamespace(text="開始，OpenAI！", language=["Chinese"], segments=[])

    def simplify(self, text):
        self.calls.append("simplify")
        return text.replace("開", "开")

    def align(self, audio, text, language):
        self.calls.append(("align", text, language))
        return [item("开", .1, .2), item("始", .2, .3), item("OpenAI", .4, .8)]


class AdapterTests(unittest.TestCase):
    def test_wav_validation_and_silence_gate(self):
        audio = q.decode_wav(wav_bytes([0] * 16000))
        self.assertTrue(q.is_near_digital_silence(audio))
        self.assertEqual(audio.duration, 1)
        self.assertFalse(q.is_near_digital_silence(q.decode_wav(wav_bytes([6] * 16000))))
        for payload in [b"bad", wav_bytes([], 16000), wav_bytes([0] * 400, 8000)]:
            with self.assertRaises(q.AdapterError):
                q.decode_wav(payload)

    def test_split_covers_audio_once_and_bounds_each_chunk(self):
        audio = q.Audio(struct.pack("<h", 10) * (16000 * 65))
        chunks = q.split_audio(audio)
        self.assertEqual(b"".join(chunk.pcm for chunk, _ in chunks), audio.pcm)
        self.assertTrue(all(chunk.duration <= 30 for chunk, _ in chunks))
        count = 0
        for chunk, offset in chunks:
            self.assertEqual(offset, count / 16000)
            count += chunk.frames

    def test_language_and_glossary(self):
        self.assertEqual(q.language_code("zh-CN"), "zh")
        self.assertEqual(q.language_code("Chinese"), "zh")
        self.assertEqual(q.language_code("auto"), "auto")
        self.assertEqual(q.glossary_hotwords("Glossary: OpenAI, 千帆，OpenAI."), ["OpenAI", "千帆"])
        with self.assertRaises(q.AdapterError):
            q.language_code("unsupported")
        result = SimpleNamespace(language=["Chinese"], segments=[{"language": "English"}])
        self.assertEqual(q.detected_language(result, "auto"), "zh")
        self.assertEqual(result.language, ["Chinese"])

    def test_malformed_multipart_reports_request_error(self):
        for content_type in ["multipart/form-data", "multipart/form-data; boundary=中", "application/json"]:
            with self.assertRaises(q.AdapterError):
                q.parse_multipart(content_type, b"invalid")

    def test_alignment_preserves_chinese_punctuation_and_english(self):
        text = "“你好，OpenAI！”"
        words = q.attach_text_to_alignment(text, [item("你", 0, .1), item("好", .1, .2), item("OpenAI", .2, .5)], 1)
        self.assertEqual("".join(word["word"] for word in words), text)
        self.assertTrue(all("probability" not in word for word in words))
        for bad in [[item("错", 0, .1)], [item("你好OpenAI", 0, 8)], [item("你好OpenAI", float("nan"), 1)]]:
            with self.assertRaises(q.AdapterError):
                q.attach_text_to_alignment(text, bad, 1)

    def test_simplify_before_alignment_and_actual_language(self):
        engine = FakeEngine()
        adapter = q.Adapter(engine)
        adapter.load()
        result = adapter.process(q.Audio(struct.pack("<h", 100) * 16000), "auto", "Glossary: OpenAI.")
        self.assertIn(("align", "开始，OpenAI！", "zh"), engine.calls)
        self.assertEqual(result["detected_language"], "zh")
        self.assertEqual(result["backend"], "qwen3-mlx")
        self.assertEqual(result["timing"]["audio_s"], 1)
        self.assertEqual("".join(w["word"] for w in result["segments"][0]["words"]), "开始，OpenAI！")

    def test_silent_audio_skips_both_models(self):
        engine = FakeEngine()
        adapter = q.Adapter(engine)
        adapter.load()
        result = adapter.process(q.Audio(b"\0\0" * 16000))
        self.assertEqual(result["segments"], [])
        self.assertEqual(engine.calls, ["load"])

    def test_cancel_after_asr_prevents_aligner_and_stale_result(self):
        engine = FakeEngine()
        adapter = q.Adapter(engine)
        adapter.load()
        engine.after_asr = adapter.cancel
        with self.assertRaises(q.Cancelled):
            adapter.process(q.Audio(struct.pack("<h", 100) * 16000))
        self.assertFalse(any(isinstance(call, tuple) and call[0] == "align" for call in engine.calls))

    def test_output_limit_is_not_saved_as_complete_transcription(self):
        engine = FakeEngine()
        engine.transcribe = lambda *args: SimpleNamespace(text="partial", generation_tokens=2048)
        adapter = q.Adapter(engine)
        adapter.load()
        with self.assertRaises(q.AdapterError) as caught:
            adapter.process(q.Audio(struct.pack("<h", 100) * 16000))
        self.assertEqual(caught.exception.kind, "output-truncated")

    def test_http_health_errors_inference_and_cancel(self):
        engine = FakeEngine()
        adapter = q.Adapter(engine)
        server = q.Server(("127.0.0.1", 0), adapter)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def request(method, path, body=None, headers=None):
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
            connection.request(method, path, body, headers or {})
            response = connection.getresponse()
            payload = json.loads(response.read())
            status = response.status
            connection.close()
            return status, payload

        try:
            self.assertEqual(request("GET", "/")[0], 503)
            adapter.load()
            status, health = request("GET", "/")
            self.assertEqual(status, 200)
            self.assertTrue(health["ready"])
            self.assertEqual(request("GET", "/", headers={"Origin": "https://example.com"})[0], 403)
            self.assertEqual(request("POST", "/inference", b"bad", {"Content-Type": "application/json"})[0], 415)
            boundary = "test-qwen-boundary"
            body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode()
                    + wav_bytes([100] * 16000) + f"\r\n--{boundary}--\r\n".encode())
            status, result = request("POST", "/inference", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
            self.assertEqual(status, 200)
            self.assertTrue(result["segments"])
            self.assertEqual(request("POST", "/cancel")[1]["cancelled"], True)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
