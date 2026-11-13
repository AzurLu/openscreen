import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

/** Opt-in local MLX helper. All paths are resolved by the runtime installer. */
export interface QwenMlxProviderConfig {
	provider: "qwen3-mlx";
	pythonPath: string;
	serverPath: string;
	asrModelPath: string;
	alignerModelPath: string;
}

/** Missing config or an explicit Whisper selection keeps the existing default. */
export function readSttProviderConfig(userDataDir: string): QwenMlxProviderConfig | null {
	let value: unknown;
	try {
		value = JSON.parse(readFileSync(path.join(userDataDir, "stt-provider.json"), "utf8"));
	} catch (error) {
		if ((error as NodeJS.ErrnoException)?.code === "ENOENT") return null;
		throw new Error(
			`Cannot read stt-provider.json: ${error instanceof Error ? error.message : String(error)}`,
		);
	}
	if (!value || typeof value !== "object" || Array.isArray(value)) {
		throw new Error("Invalid STT provider in stt-provider.json");
	}
	const config = value as Record<string, unknown>;
	if (config.provider === "whisper") return null;
	if (config.provider !== "qwen3-mlx") {
		throw new Error("Unsupported STT provider in stt-provider.json");
	}
	if (process.platform !== "darwin" || process.arch !== "arm64") {
		throw new Error("Qwen MLX transcription requires an Apple Silicon Mac");
	}
	const requiredPath = (key: string): string => {
		const candidate = config[key];
		if (typeof candidate !== "string" || !path.isAbsolute(candidate) || !existsSync(candidate)) {
			throw new Error(`Qwen STT ${key} must be an existing absolute path`);
		}
		return candidate;
	};
	return {
		provider: "qwen3-mlx",
		pythonPath: requiredPath("pythonPath"),
		serverPath: requiredPath("serverPath"),
		asrModelPath: requiredPath("asrModelPath"),
		alignerModelPath: requiredPath("alignerModelPath"),
	};
}
