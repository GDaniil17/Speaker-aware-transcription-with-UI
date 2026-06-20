"""Export transcript segments to txt / json / srt / vtt."""

from __future__ import annotations

import json
from collections.abc import Sequence

from speaker_transcribe.pipeline import TranscriptSegment


def _ts(t: float, sep: str = ",") -> str:
    """Format seconds as HH:MM:SS,mmm (srt) or HH:MM:SS.mmm (vtt)."""
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms == 1000:
        s, ms = s + 1, 0
    return f"{h:02}:{m:02}:{s:02}{sep}{ms:03}"


def to_txt(segments: Sequence[TranscriptSegment], include_speakers: bool = True) -> str:
    """Readable transcript, optionally grouped by consecutive speaker turns."""
    lines: list[str] = []
    prev_speaker = None
    for seg in segments:
        if include_speakers and seg.speaker != prev_speaker:
            lines.append(f"\n[{seg.speaker}]")
            prev_speaker = seg.speaker
        lines.append(seg.text)
    return "\n".join(lines).strip() + "\n"


def to_json(segments: Sequence[TranscriptSegment], include_speakers: bool = True) -> str:
    data = []
    for seg in segments:
        entry = {"start": round(seg.start, 3), "end": round(seg.end, 3)}
        if include_speakers:
            entry["speaker"] = seg.speaker
        entry["text"] = seg.text
        data.append(entry)
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def to_srt(segments: Sequence[TranscriptSegment], include_speakers: bool = True) -> str:
    blocks: list[str] = []
    for i, seg in enumerate(segments, start=1):
        prefix = f"[{seg.speaker}] " if include_speakers else ""
        blocks.append(f"{i}\n{_ts(seg.start)} --> {_ts(seg.end)}\n{prefix}{seg.text}\n")
    return "\n".join(blocks)


def to_vtt(segments: Sequence[TranscriptSegment], include_speakers: bool = True) -> str:
    blocks = ["WEBVTT\n"]
    for seg in segments:
        voice = f"<v {seg.speaker}>" if include_speakers else ""
        blocks.append(f"{_ts(seg.start, '.')} --> {_ts(seg.end, '.')}\n{voice}{seg.text}\n")
    return "\n".join(blocks)


EXPORTERS = {
    "txt": to_txt,
    "json": to_json,
    "srt": to_srt,
    "vtt": to_vtt,
}


def export(segments: Sequence[TranscriptSegment], fmt: str, include_speakers: bool = True) -> str:
    try:
        return EXPORTERS[fmt](segments, include_speakers)
    except KeyError:
        raise ValueError(f"Unknown format: {fmt!r}. Choose from {list(EXPORTERS)}.")
