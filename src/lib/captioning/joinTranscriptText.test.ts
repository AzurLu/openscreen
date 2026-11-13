import { describe, expect, it } from "vitest";
import { joinTranscriptText } from "./joinTranscriptText";

const cases: Array<{ label: string; parts: string[]; expected: string }> = [
	{
		label: "attached Chinese punctuation",
		parts: ["今", "天，", "测", "试。"],
		expected: "今天，测试。",
	},
	{
		label: "separate Chinese punctuation",
		parts: ["今", "天", "，", "测", "试", "。"],
		expected: "今天，测试。",
	},
	{
		label: "mixed English model names",
		parts: ["使用", "Qwen3", "ASR", "模型"],
		expected: "使用 Qwen3 ASR 模型",
	},
	{ label: "number boundaries", parts: ["第", "3", "次", "测试"], expected: "第 3 次测试" },
	{ label: "corner quotes", parts: ["「", "中", "文", "」", "测", "试"], expected: "「中文」测试" },
	{
		label: "attached curly quotes",
		parts: ["你", "说：“", "好", "的。", "”"],
		expected: "你说：“好的。”",
	},
	{
		label: "separate curly quotes",
		parts: ["他", "说", "：", "“", "你", "好", "”", "。"],
		expected: "他说：“你好”。",
	},
	{
		label: "Chinese parentheses",
		parts: ["今天", "（", "下雨", "）", "。"],
		expected: "今天（下雨）。",
	},
	{
		label: "quoted English in Chinese",
		parts: ["使用", "“OpenScreen”", "录制"],
		expected: "使用“OpenScreen”录制",
	},
	{
		label: "Chinese comma after English",
		parts: ["OpenScreen，", "今天"],
		expected: "OpenScreen，今天",
	},
	{
		label: "consecutive English words",
		parts: ["中文", "Hello", "world"],
		expected: "中文 Hello world",
	},
	{ label: "supplementary Han character", parts: ["𠀀", "文"], expected: "𠀀文" },
	{ label: "English sentence unchanged", parts: ["Hello,", "world!"], expected: "Hello, world!" },
	{
		label: "English separate punctuation unchanged",
		parts: ["Hello", ",", "world"],
		expected: "Hello , world",
	},
	{
		label: "English whitespace unchanged",
		parts: [" Hello", "", "world "],
		expected: " Hello  world ",
	},
	{ label: "Korean spacing unchanged", parts: ["안녕", "하세요"], expected: "안녕 하세요" },
];

describe("joinTranscriptText", () => {
	it.each(cases)("$label", ({ parts, expected }) => {
		expect(joinTranscriptText(parts)).toBe(expected);
	});

	it("handles empty input and leaves source fragments untouched", () => {
		const parts = Object.freeze([" 今 ", "", "天。 "]);
		expect(joinTranscriptText(parts)).toBe("今天。");
		expect(parts).toEqual([" 今 ", "", "天。 "]);
		expect(joinTranscriptText([])).toBe("");
	});
});
