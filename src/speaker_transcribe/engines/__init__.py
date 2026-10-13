"""ASR engines. `transcribe` dispatches to the one a run's settings select.

Each engine module imports its model library lazily, inside `transcribe`, so an
engine that is not installed costs nothing until it is picked.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from speaker_transcribe.config import Settings
from speaker_transcribe.engines.common import ProgressBarFn, ProgressFn
from speaker_transcribe.transcript import TranscriptSegment


def transcribe(
    wav_path: Path,
    settings: Settings,
    diar: list[dict],
    sp_map: dict,
    progress: ProgressFn | None,
    on_progress: ProgressBarFn | None = None,
) -> Iterator[TranscriptSegment]:
    """Dispatch to the configured transcription engine."""
    if settings.engine == "parakeet":
        from speaker_transcribe.engines import parakeet as engine
    elif settings.engine == "qwen3-asr":
        from speaker_transcribe.engines import qwen3 as engine
    elif settings.engine == "sensevoice":
        from speaker_transcribe.engines import sensevoice as engine
    elif settings.engine == "gigaam":
        from speaker_transcribe.engines import gigaam as engine
    else:
        from speaker_transcribe.engines import whisper

        if settings.backend == "mlx":
            run = whisper.transcribe_mlx
        else:
            run = whisper.transcribe_faster_whisper
        yield from run(wav_path, settings, diar, sp_map, progress, on_progress)
        return
    yield from engine.transcribe(wav_path, settings, diar, sp_map, progress, on_progress)
