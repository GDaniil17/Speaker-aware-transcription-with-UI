"""NVIDIA Parakeet: parakeet-mlx on Apple Silicon, NeMo elsewhere."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from speaker_transcribe.catalog import PARAKEET_REPOS
from speaker_transcribe.config import Settings
from speaker_transcribe.engines.common import (
    ProgressBarFn,
    ProgressFn,
    cached_model,
    log,
    module_available,
)
from speaker_transcribe.hub import ensure_repo_downloaded
from speaker_transcribe.runtime import is_apple_silicon
from speaker_transcribe.transcript import TranscriptSegment, emit_segment


def transcribe(
    wav_path: Path,
    settings: Settings,
    diar: list[dict],
    sp_map: dict,
    progress: ProgressFn | None,
    on_progress: ProgressBarFn | None = None,
) -> Iterator[TranscriptSegment]:
    """Transcribe with NVIDIA Parakeet (MLX on Apple Silicon, else NeMo).

    Parakeet auto-detects its language (25 European languages); settings.language
    is ignored. Both backends yield sentence-level timestamps.
    """
    use_mlx = is_apple_silicon() and module_available("parakeet_mlx")
    log(progress, "Loading Parakeet model…")
    if on_progress:
        on_progress("Transcribing (Parakeet, no live progress)", 0.0, None)

    if use_mlx:
        from parakeet_mlx import from_pretrained

        repo = PARAKEET_REPOS["mlx"]
        ensure_repo_downloaded(repo, on_progress)
        log(progress, f"Transcribing with Parakeet MLX ({repo})…")
        model = cached_model(("parakeet-mlx", repo), lambda: from_pretrained(repo))
        result = model.transcribe(
            str(wav_path),
            chunk_duration=120.0,
            overlap_duration=15.0,
        )
        rows = [(s.start, s.end, s.text) for s in result.sentences]
    else:
        if not module_available("nemo"):
            raise RuntimeError(
                "Parakeet needs NeMo (or parakeet-mlx on Apple Silicon). "
                "Install it with: uv sync --inexact --extra parakeet"
            )
        import nemo.collections.asr as nemo_asr  # type: ignore

        repo = PARAKEET_REPOS["nemo"]
        ensure_repo_downloaded(repo, on_progress)
        log(progress, f"Transcribing with Parakeet NeMo ({repo})…")
        model = cached_model(
            ("parakeet-nemo", repo),
            lambda: nemo_asr.models.ASRModel.from_pretrained(repo),
        )
        out = model.transcribe([str(wav_path)], timestamps=True)
        first = out[0]
        stamp = first.timestamp if hasattr(first, "timestamp") else first["timestamp"]
        segs = stamp["segment"]
        rows = [(s["start"], s["end"], s.get("segment", s.get("text", ""))) for s in segs]

    total = rows[-1][1] if rows else (settings.duration or None)
    for start, end, text in rows:
        if on_progress and total:
            on_progress("Transcribing", min(end / total, 1.0), 0.0)
        seg = emit_segment(start, end, text, diar, sp_map)
        if seg:
            yield seg

    if on_progress:
        on_progress("Transcribing", 1.0, 0.0)
