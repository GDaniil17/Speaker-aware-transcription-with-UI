"""Sber GigaAM v3 and v2 (Russian) on ONNX Runtime via onnx-asr."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from speaker_transcribe.catalog import (
    GIGAAM_FILES,
    GIGAAM_MODELS,
    ONNX_QUANTIZATION,
    VAD_FILES,
    VAD_REPO,
    gigaam_repo_for,
)
from speaker_transcribe.config import Settings
from speaker_transcribe.engines.common import ProgressBarFn, ProgressFn, log, module_available
from speaker_transcribe.engines.onnx import transcribe_onnx
from speaker_transcribe.hub import ensure_repo_downloaded
from speaker_transcribe.transcript import TranscriptSegment


def transcribe(
    wav_path: Path,
    settings: Settings,
    diar: list[dict],
    sp_map: dict,
    progress: ProgressFn | None,
    on_progress: ProgressBarFn | None = None,
) -> Iterator[TranscriptSegment]:
    """Transcribe Russian audio with Sber GigaAM (v3 or v2).

    The v3 e2e models write punctuation and capitals; the others lowercase
    words only.
    """
    if not module_available("onnx_asr"):
        raise RuntimeError("GigaAM needs the onnx-asr package. Install it with: uv sync")
    model_name = settings.whisper_model
    if model_name not in GIGAAM_MODELS:
        model_name = next(iter(GIGAAM_MODELS))
    log(progress, f"Loading GigaAM model ({model_name})…")
    ensure_repo_downloaded(gigaam_repo_for(model_name), on_progress, GIGAAM_FILES[model_name])
    ensure_repo_downloaded(VAD_REPO, on_progress, VAD_FILES)
    yield from transcribe_onnx(
        model_name, ONNX_QUANTIZATION, "GigaAM", wav_path, diar, sp_map, progress, on_progress
    )
