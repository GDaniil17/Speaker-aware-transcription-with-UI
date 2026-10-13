"""End to end: prepare audio, diarize, transcribe, attribute speakers."""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

from speaker_transcribe.audio import prepare_audio, slice_wav, wav_duration
from speaker_transcribe.chunking import Unit, run_with_fallback
from speaker_transcribe.config import Settings
from speaker_transcribe.engines.common import ProgressBarFn, ProgressFn, log
from speaker_transcribe.transcript import TranscriptSegment
from speaker_transcribe.worker import transcription_runner

__all__ = ["TranscriptSegment", "run_pipeline"]


def _shift_diar(diar: list[dict], offset: float) -> list[dict]:
    """Diarization turns moved into a chunk's own timeline.

    Speaker attribution compares transcript times against diarization times, so
    a chunk that starts at 5:00 needs turns expressed from its own zero.
    """
    if not offset:
        return diar
    return [
        {"start": turn["start"] - offset, "end": turn["end"] - offset, "speaker": turn["speaker"]}
        for turn in diar
    ]


def _unit_progress(
    unit: Unit,
    total: float,
    on_progress: ProgressBarFn | None,
) -> ProgressBarFn | None:
    """Map one chunk's progress onto the whole file's progress bar."""
    if on_progress is None or unit.chunk_size is None:
        return on_progress  # one call over everything — its progress is the run's

    def report(stage: str, frac: float, eta: float | None) -> None:
        if not stage.startswith("Transcribing"):
            on_progress(stage, frac, eta)  # downloading a model — its own scale
            return
        # Engines that return everything in one call report no progress inside a
        # chunk; the chunk boundaries themselves still move the bar along.
        inside = 0.0 if "no live progress" in stage else min(max(frac, 0.0), 1.0)
        elapsed = unit.start + inside * unit.duration
        done = min(elapsed / total, 1.0) if total > 0 else 0.0
        on_progress(f"Transcribing · {unit.short_label}", done, None)

    return report


def _transcribe_all(
    wav_path: Path,
    settings: Settings,
    diar: list[dict],
    sp_map: dict,
    progress: ProgressFn | None,
    on_progress: ProgressBarFn | None = None,
) -> list[TranscriptSegment]:
    """Transcribe the prepared WAV, cutting it into chunks whenever it stalls."""
    total = wav_duration(wav_path)
    ladder = tuple(settings.chunk_ladder) if settings.chunk_fallback else ()
    chunk_dir = wav_path.parent / "chunks"

    with transcription_runner(settings.stall_timeout) as run_job:

        def run_unit(unit: Unit) -> list[TranscriptSegment]:
            log(progress, f"Transcribing {unit.label}…")
            whole = unit.chunk_size is None and unit.start <= 0 and unit.end >= total - 0.05
            piece = (
                wav_path
                if whole
                else slice_wav(
                    wav_path,
                    chunk_dir / f"chunk_{int(unit.start)}_{int(unit.end)}.wav",
                    unit.start,
                    unit.end,
                )
            )
            try:
                rows = run_job(
                    piece,
                    replace(settings, start=0.0, duration=unit.duration),
                    _shift_diar(diar, unit.start),
                    sp_map,
                    progress,
                    _unit_progress(unit, total, on_progress),
                )
            finally:
                if not whole:
                    piece.unlink(missing_ok=True)
            return [
                TranscriptSegment(start + unit.start, end + unit.start, speaker, text)
                for start, end, speaker, text in rows
            ]

        segments = run_with_fallback(
            total,
            ladder,
            run_unit,
            log=progress,
            whole_file_first=settings.try_whole_file_first,
            skip_failed=settings.skip_failed_chunks,
        )

    segments.sort(key=lambda seg: seg.start)
    if on_progress:
        on_progress("Transcribing", 1.0, 0.0)
    return segments


def run_pipeline(
    settings: Settings,
    progress: ProgressFn | None = None,
    on_progress: ProgressBarFn | None = None,
) -> list[TranscriptSegment]:
    """End-to-end: prepare audio, diarize, transcribe, attribute speakers."""
    with tempfile.TemporaryDirectory(prefix="spk_") as tmp:
        wav_path = Path(tmp) / "input.wav"
        log(progress, "Preparing audio…")
        prepare_audio(
            settings.input_path,
            wav_path,
            start=settings.start,
            duration=settings.duration,
        )

        if settings.diarize:
            from speaker_transcribe.diarization import diarize

            diar, sp_map = diarize(wav_path, settings, progress, on_progress)
        else:
            log(progress, "Skipping diarization — transcription only.")
            diar, sp_map = [], {}
        result = _transcribe_all(wav_path, settings, diar, sp_map, progress, on_progress)

        if not settings.diarize:
            for seg in result:
                seg.speaker = "Speaker 1"

    if settings.start:
        for seg in result:
            seg.start += settings.start
            seg.end += settings.start

    log(progress, f"Done — {len(result)} transcript segment(s).")
    return result
