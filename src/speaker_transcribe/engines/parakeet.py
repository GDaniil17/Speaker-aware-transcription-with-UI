"""NVIDIA Parakeet TDT 0.6B v3 on the fastest backend installed.

parakeet-mlx (Apple Silicon GPU) or NeMo (CUDA) when their optional extras are
installed, otherwise onnx-asr on the CPU, which the core install already has.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from speaker_transcribe.catalog import (
    ONNX_QUANTIZATION,
    PARAKEET_ONNX_FILES,
    PARAKEET_ONNX_MODEL,
    PARAKEET_REPOS,
    VAD_FILES,
    VAD_REPO,
)
from speaker_transcribe.config import Settings
from speaker_transcribe.engines.common import ProgressBarFn, ProgressFn, cached_model, log
from speaker_transcribe.engines.onnx import transcribe_onnx
from speaker_transcribe.hub import ensure_repo_downloaded
from speaker_transcribe.runtime import PARAKEET_BACKEND_LABELS, parakeet_backend
from speaker_transcribe.transcript import TranscriptSegment, emit_segment


def transcribe(
    wav_path: Path,
    settings: Settings,
    diar: list[dict],
    sp_map: dict,
    progress: ProgressFn | None,
    on_progress: ProgressBarFn | None = None,
) -> Iterator[TranscriptSegment]:
    """Transcribe with NVIDIA Parakeet.

    Parakeet detects the language itself (25 European languages), so
    settings.language is ignored. The ONNX backend gives word timestamps, so
    speaker changes are split per word; MLX and NeMo give sentence timestamps.
    """
    backend = parakeet_backend()
    if backend is None:
        raise RuntimeError("Parakeet needs the onnx-asr package. Install it with: uv sync")
    log(progress, f"Loading Parakeet model ({PARAKEET_BACKEND_LABELS[backend]})…")

    if backend == "onnx":
        ensure_repo_downloaded(PARAKEET_REPOS["onnx"], on_progress, PARAKEET_ONNX_FILES)
        ensure_repo_downloaded(VAD_REPO, on_progress, VAD_FILES)
        yield from transcribe_onnx(
            PARAKEET_ONNX_MODEL,
            ONNX_QUANTIZATION,
            "Parakeet",
            wav_path,
            diar,
            sp_map,
            progress,
            on_progress,
        )
        return

    if on_progress:
        on_progress("Transcribing (Parakeet, no live progress)", 0.0, None)
    repo = PARAKEET_REPOS[backend]
    ensure_repo_downloaded(repo, on_progress)
    log(progress, f"Transcribing with Parakeet ({repo})…")
    if backend == "mlx":
        from parakeet_mlx import from_pretrained

        model = cached_model(("parakeet-mlx", repo), lambda: from_pretrained(repo))
        result = model.transcribe(str(wav_path), chunk_duration=120.0, overlap_duration=15.0)
        rows = [(s.start, s.end, s.text) for s in result.sentences]
    else:
        import nemo.collections.asr as nemo_asr  # type: ignore

        model = cached_model(
            ("parakeet-nemo", repo),
            lambda: nemo_asr.models.ASRModel.from_pretrained(repo),
        )
        out = model.transcribe([str(wav_path)], timestamps=True)
        first = out[0]
        stamp = first.timestamp if hasattr(first, "timestamp") else first["timestamp"]
        rows = [
            (s["start"], s["end"], s.get("segment", s.get("text", ""))) for s in stamp["segment"]
        ]

    total = rows[-1][1] if rows else (settings.duration or None)
    for start, end, text in rows:
        if on_progress and total:
            on_progress("Transcribing", min(end / total, 1.0), 0.0)
        seg = emit_segment(start, end, text, diar, sp_map)
        if seg:
            yield seg

    if on_progress:
        on_progress("Transcribing", 1.0, 0.0)
