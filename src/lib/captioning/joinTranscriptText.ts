const CJK = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]/u;
const CLOSING = /^[,.;:!?%。，、；：！？…）)\]}>》」』】〕”’]/u;
const TRAILING = /[,.;:!?%。，、；：！？…）)\]}>》」』】〕”’]+$/u;
const OPENING = /[([<{《「『【〔（“‘]$/u;
const CJK_OPENING = /^[《「『【〔（“‘]/u;
const CJK_CLOSING = /[。，、；：！？…）》」』】〕”’]$/u;

/**
 * Join transcript fragments for display without spacing out aligned CJK characters.
 * Preserve the existing space join exactly when there is no Han or kana text;
 * mixed-language boundaries still separate Latin words and numbers. This only
 * formats text: callers keep their original tokens, grouping and timestamps.
 */
export function joinTranscriptText(parts: readonly string[]): string {
	if (!parts.some((part) => CJK.test(part))) return parts.join(" ");
	return parts
		.map((part) => part.trim())
		.filter(Boolean)
		.reduce((text, next) => {
			if (!text) return next;
			if (CLOSING.test(next) || OPENING.test(text) || CJK_OPENING.test(next)) {
				return text + next;
			}
			const last = Array.from(text.replace(TRAILING, "")).pop() ?? "";
			const first = Array.from(next)[0] ?? "";
			return CJK.test(first) && (CJK.test(last) || CJK_CLOSING.test(text))
				? text + next
				: `${text} ${next}`;
		}, "");
}
