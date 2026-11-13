import { describe, expect, it } from "vitest";

import {
	dedupeAdjacentCaptionRepeats,
	finalizeCaptionSegmentsForPlayback,
	groupPhraseCaptionSegmentsIntoLines,
	groupTimedCaptionWordsIntoLines,
	splitMergedCaptionsByWordBounds,
} from "./annotationsFromCaptions";

describe("groupPhraseCaptionSegmentsIntoLines", () => {
	it("preserves phrase boundaries when formatting phrase-timestamp captions", () => {
		const lines = groupPhraseCaptionSegmentsIntoLines(
			[
				{ startSec: 0, endSec: 0.5, text: "alpha beta" },
				{ startSec: 0.62, endSec: 1.6, text: "gamma delta" },
			],
			2,
			2,
		);

		expect(lines).toHaveLength(2);
		expect(lines[0]).toMatchObject({ text: "alpha beta", startSec: 0 });
		expect(lines[1]).toMatchObject({ text: "gamma delta", startSec: 0.62 });
		expect(lines[0]!.endSec).toBeLessThanOrEqual(0.62);
	});

	it("slices a single merged phrase into timed caption lines by word bounds", () => {
		const lines = groupPhraseCaptionSegmentsIntoLines(
			[{ startSec: 0, endSec: 1, text: "alpha beta gamma delta" }],
			2,
			2,
		);

		expect(lines).toHaveLength(2);
		expect(lines[0]).toMatchObject({
			startSec: 0,
			endSec: 0.5,
			text: "alpha beta",
		});
		expect(lines[1]).toMatchObject({
			startSec: 0.5,
			endSec: 1,
			text: "gamma delta",
		});
	});
});

describe("groupTimedCaptionWordsIntoLines", () => {
	it("preserves empty timeline space when word timestamps contain a real pause", () => {
		const lines = groupTimedCaptionWordsIntoLines(
			[
				{ startSec: 0, endSec: 0.12, text: "first" },
				{ startSec: 0.13, endSec: 0.28, text: "caption" },
				{ startSec: 0.7, endSec: 0.83, text: "second" },
				{ startSec: 0.84, endSec: 0.98, text: "caption" },
			],
			2,
			2,
		);

		expect(lines).toHaveLength(2);
		expect(lines[0]).toMatchObject({ startSec: 0, endSec: 0.28, text: "first caption" });
		expect(lines[1]).toMatchObject({ startSec: 0.7, endSec: 0.98, text: "second caption" });
	});

	it("preserves repeated words instead of collapsing them into one token", () => {
		const lines = groupTimedCaptionWordsIntoLines(
			[
				{ startSec: 0, endSec: 0.12, text: "I" },
				{ startSec: 0.13, endSec: 0.25, text: "I" },
			],
			2,
			2,
		);

		expect(lines).toHaveLength(1);
		expect(lines[0]).toMatchObject({ text: "I I" });
	});
});

describe("dedupeAdjacentCaptionRepeats", () => {
	it("merges the same line repeated across a chunk boundary", () => {
		const lines = dedupeAdjacentCaptionRepeats([
			{ startSec: 0, endSec: 1, text: "hello there" },
			{ startSec: 0.9, endSec: 1.8, text: "Hello there." },
		]);

		expect(lines).toHaveLength(1);
		expect(lines[0]).toMatchObject({ startSec: 0, endSec: 1.8 });
	});

	it("keeps the same line spoken again after a real silence", () => {
		const lines = dedupeAdjacentCaptionRepeats([
			{ startSec: 0, endSec: 1, text: "hello there" },
			{ startSec: 8, endSec: 9, text: "hello there" },
		]);

		expect(lines).toHaveLength(2);
	});
});

describe("finalizeCaptionSegmentsForPlayback", () => {
	it("ends a line where the next one starts so two never show at once", () => {
		const lines = finalizeCaptionSegmentsForPlayback([
			{ startSec: 0, endSec: 1.5, text: "first" },
			{ startSec: 1, endSec: 2, text: "second" },
		]);

		expect(lines[0]!.endSec).toBeCloseTo(1, 5);
		expect(lines[1]).toMatchObject({ startSec: 1, endSec: 2 });
	});
});

describe("CJK caption display", () => {
	it("keeps all 55 aligned tokens and the same line boundaries as English", () => {
		const words = Array.from({ length: 55 }, (_, i) => ({
			text: ["今", "天", "测", "试", "中", "文", "。 "][i % 7],
			startSec: i * 0.25,
			endSec: (i + 1) * 0.25,
		}));
		const original = structuredClone(words);
		const chinese = groupTimedCaptionWordsIntoLines(words, 2, 7);
		const english = groupTimedCaptionWordsIntoLines(
			words.map((word, i) => ({ ...word, text: `word${i}` })),
			2,
			7,
		);
		expect(chinese).toHaveLength(8);
		expect(chinese.map(({ startSec, endSec }) => [startSec, endSec])).toEqual(
			english.map(({ startSec, endSec }) => [startSec, endSec]),
		);
		expect(chinese.map((line) => line.text).join("")).toBe(
			words.map((word) => word.text.trim()).join(""),
		);
		expect(words).toEqual(original);
	});

	it("removes CJK display gaps without changing phrase splitting or interpolation", () => {
		expect(
			splitMergedCaptionsByWordBounds([{ startSec: 0, endSec: 4, text: "今 天 测 试" }], 2, 2),
		).toEqual([
			{ startSec: 0, endSec: 2, text: "今天" },
			{ startSec: 2, endSec: 4, text: "测试" },
		]);
		expect(
			splitMergedCaptionsByWordBounds([{ startSec: 0, endSec: 4, text: "今 天 测 试" }], 2, 7),
		).toEqual([{ startSec: 0, endSec: 4, text: "今天测试" }]);
	});
});
