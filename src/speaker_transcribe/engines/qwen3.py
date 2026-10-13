"""Qwen3-ASR-1.7B with the Qwen3 forced aligner for word timestamps."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

from speaker_transcribe.audio import slice_wav, wav_duration
from speaker_transcribe.catalog import QWEN3_ALIGNER_REPO, QWEN3_ASR_LANGUAGES, QWEN3_ASR_REPO
from speaker_transcribe.chunking import split_region
from speaker_transcribe.config import Settings
from speaker_transcribe.engines.common import (
    ProgressBarFn,
    ProgressFn,
    cached_model,
    log,
    module_available,
)
from speaker_transcribe.hub import ensure_repo_downloaded
from speaker_transcribe.transcript import TranscriptSegment, emit_segment, group_words


def transcribe(
    wav_path: Path,
    settings: Settings,
    diar: list[dict],
    sp_map: dict,
    progress: ProgressFn | None,
    on_progress: ProgressBarFn | None = None,
) -> Iterator[TranscriptSegment]:
    """Transcribe with Qwen3-ASR-1.7B + its forced aligner for word timestamps.

    The aligner handles up to ~5 minutes per call, so long audio is split into
    fixed windows; word timestamps are offset back to absolute time and grouped
    into sentence-like segments before speaker attribution.
    """
    if not module_available("qwen_asr"):
        raise RuntimeError(
            "Qwen3-ASR needs the qwen-asr package. "
            "Install it with: uv sync --inexact --extra qwen3-asr"
        )
    from qwen_asr import Qwen3ASRModel

    log(progress, "Loading Qwen3-ASR model…")
    ensure_repo_downloaded(QWEN3_ASR_REPO, on_progress)
    ensure_repo_downloaded(QWEN3_ALIGNER_REPO, on_progress)
    if on_progress:
        on_progress("Transcribing (Qwen3-ASR, no live progress)", 0.0, None)
    model = cached_model(
        ("qwen3-asr", QWEN3_ASR_REPO, QWEN3_ALIGNER_REPO),
        lambda: Qwen3ASRModel.from_pretrained(QWEN3_ASR_REPO, forced_aligner=QWEN3_ALIGNER_REPO),
    )

    # Qwen wants the language *name* (e.g. "English"); None = auto-detect.
    language = QWEN3_ASR_LANGUAGES.get(settings.language) if settings.language else None

    words: list[tuple[float, float, str]] = []
    with tempfile.TemporaryDirectory(prefix="qwen3_") as tmp:
        chunks = list(_wav_windows(wav_path, Path(tmp), window_seconds=240))
        for i, (chunk_path, chunk_start) in enumerate(chunks):
            if on_progress and chunks:
                on_progress("Transcribing", min(i / len(chunks), 1.0), None)
            res = model.transcribe(
                audio=str(chunk_path), language=language, return_time_stamps=True
            )
            if not res:
                continue
            for ts in res[0].time_stamps or []:
                words.append(
                    (
                        chunk_start + float(ts.start_time),
                        chunk_start + float(ts.end_time),
                        str(ts.text),
                    )
                )

    for start, end, text in group_words(words):
        seg = emit_segment(start, end, text, diar, sp_map)
        if seg:
            yield seg

    if on_progress:
        on_progress("Transcribing", 1.0, 0.0)


def _wav_windows(
    wav_path: Path,
    out_dir: Path,
    window_seconds: int = 240,
) -> Iterator[tuple[Path, float]]:
    """Split a WAV into fixed windows, yielding (path, start_seconds)."""
    spans = split_region(0.0, wav_duration(wav_path), float(window_seconds))
    for index, (start, end) in enumerate(spans):
        yield slice_wav(wav_path, out_dir / f"window_{index:04d}.wav", start, end), start
