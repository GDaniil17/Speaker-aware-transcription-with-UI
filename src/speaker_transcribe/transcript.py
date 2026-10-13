"""Transcript segments and how they are attributed to speakers.

Every engine produces timed text; these helpers turn it into
`TranscriptSegment`s labelled with the diarization speaker it overlaps most.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class TranscriptSegment:
    """One transcribed chunk attributed to a speaker."""

    start: float
    end: float
    speaker: str
    text: str


def speaker_for(seg_start: float, seg_end: float, diar: list[dict], sp_map: dict) -> str:
    """Attribute a transcription segment to the speaker with the most overlap."""
    best_speaker = "Unknown"
    best_overlap = 0.0
    for ds in diar:
        overlap = max(0.0, min(seg_end, ds["end"]) - max(seg_start, ds["start"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = sp_map[ds["speaker"]]
    return best_speaker


def emit_segment(
    start: float,
    end: float,
    text: str,
    diar: list[dict],
    sp_map: dict,
) -> TranscriptSegment | None:
    """Build a speaker-attributed segment, or None when the text is empty."""
    text = (text or "").strip()
    if not text:
        return None
    return TranscriptSegment(start, end, speaker_for(start, end, diar, sp_map), text)


def split_by_speaker(
    start: float,
    end: float,
    text: str,
    words: list[tuple[float, float, str]],
    diar: list[dict],
    sp_map: dict,
) -> Iterator[TranscriptSegment]:
    """One recognised piece, split where diarization says the speaker changes.

    VAD cuts on silence, not on speaker turns, so a 20 s piece can hold a quick
    reply from the other person. Each word goes to the speaker it overlaps most;
    words with no overlap stay with the piece's main speaker.
    """
    main = speaker_for(start, end, diar, sp_map)
    if not diar or not words:
        seg = emit_segment(start, end, text, diar, sp_map)
        if seg:
            yield seg
        return
    run: list[tuple[float, float, str]] = []
    run_speaker = None
    for w_start, w_end, word in words:
        speaker = speaker_for(w_start, w_end, diar, sp_map)
        speaker = main if speaker == "Unknown" else speaker
        if run and speaker != run_speaker:
            yield TranscriptSegment(run[0][0], run[-1][1], run_speaker, " ".join(w[2] for w in run))
            run = []
        run.append((w_start, w_end, word))
        run_speaker = speaker
    if run:
        yield TranscriptSegment(run[0][0], run[-1][1], run_speaker, " ".join(w[2] for w in run))


def join_tokens(tokens: list[str]) -> str:
    """Join ASR tokens into text — spaces for word tokens, none for CJK characters."""
    word_like = any(len(t.strip()) > 1 and t.strip().isalpha() for t in tokens)
    sep = " " if word_like else ""
    return sep.join(t.strip() for t in tokens if t.strip()).strip()


def group_words(
    words: list[tuple[float, float, str]],
    gap: float = 0.6,
    max_chars: int = 240,
) -> Iterator[tuple[float, float, str]]:
    """Merge word/char timestamps into readable segments.

    Breaks on sentence-ending punctuation, a silence gap longer than `gap`
    seconds, or when a segment grows past `max_chars` characters.
    """
    cur: list[tuple[float, float, str]] = []

    def flush() -> Iterator[tuple[float, float, str]]:
        if cur:
            yield cur[0][0], cur[-1][1], join_tokens([w[2] for w in cur])

    for i, w in enumerate(words):
        cur.append(w)
        text_so_far = "".join(x[2] for x in cur)
        ends_sentence = w[2].strip().endswith((".", "?", "!", "。", "？", "！", "…"))
        next_gap = words[i + 1][0] - w[1] if i + 1 < len(words) else 0.0
        if ends_sentence or next_gap > gap or len(text_so_far) >= max_chars:
            yield from flush()
            cur = []
    yield from flush()
