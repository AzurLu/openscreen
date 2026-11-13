import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { type QwenMlxProviderConfig, readSttProviderConfig } from "./providerConfig";

describe("local STT provider configuration", () => {
	let dir: string;
	let config: QwenMlxProviderConfig;
	const platform = Object.getOwnPropertyDescriptor(process, "platform");
	const arch = Object.getOwnPropertyDescriptor(process, "arch");
	const writeConfig = (value: unknown) =>
		writeFileSync(path.join(dir, "stt-provider.json"), JSON.stringify(value));
	beforeEach(() => {
		Object.defineProperty(process, "platform", { value: "darwin", configurable: true });
		Object.defineProperty(process, "arch", { value: "arm64", configurable: true });
		dir = mkdtempSync(path.join(tmpdir(), "stt-provider-"));
		config = {
			provider: "qwen3-mlx",
			pythonPath: path.join(dir, "python"),
			serverPath: path.join(dir, "qwen_server.py"),
			asrModelPath: path.join(dir, "asr"),
			alignerModelPath: path.join(dir, "aligner"),
		};
		writeFileSync(config.pythonPath, "python");
		writeFileSync(config.serverPath, "server");
		mkdirSync(config.asrModelPath);
		mkdirSync(config.alignerModelPath);
	});
	afterEach(() => {
		if (platform) Object.defineProperty(process, "platform", platform);
		if (arch) Object.defineProperty(process, "arch", arch);
		rmSync(dir, { recursive: true, force: true });
	});
	it("keeps Whisper when no config exists, and ignores Qwen paths when Whisper is selected", () => {
		expect(readSttProviderConfig(dir)).toBeNull();
		writeConfig({ provider: "whisper", pythonPath: "not-installed" });
		Object.defineProperty(process, "platform", { value: "linux", configurable: true });
		expect(readSttProviderConfig(dir)).toBeNull();
	});
	it("loads the four explicit local runtime/model paths", () => {
		writeConfig(config);
		expect(readSttProviderConfig(dir)).toEqual(config);
	});
	it("reports malformed config and unsupported providers instead of silently using Whisper", () => {
		writeFileSync(path.join(dir, "stt-provider.json"), "{");
		expect(() => readSttProviderConfig(dir)).toThrow(/Cannot read stt-provider.json/);
		writeConfig({ provider: "remote" });
		expect(() => readSttProviderConfig(dir)).toThrow(/Unsupported STT provider/);
		writeConfig(null);
		expect(() => readSttProviderConfig(dir)).toThrow(/Invalid STT provider/);
	});
	it.each([
		"pythonPath",
		"serverPath",
		"asrModelPath",
		"alignerModelPath",
	] as const)("requires an existing absolute %s", (key) => {
		for (const bad of [undefined, "relative/path", path.join(dir, "missing")]) {
			writeConfig({ ...config, [key]: bad });
			expect(() => readSttProviderConfig(dir)).toThrow(new RegExp(key));
		}
	});
	it.each([
		["linux", "arm64"],
		["darwin", "x64"],
	])("rejects MLX on %s/%s", (os, cpu) => {
		writeConfig(config);
		Object.defineProperty(process, "platform", { value: os, configurable: true });
		Object.defineProperty(process, "arch", { value: cpu, configurable: true });
		expect(() => readSttProviderConfig(dir)).toThrow(/Apple Silicon Mac/);
	});
});
