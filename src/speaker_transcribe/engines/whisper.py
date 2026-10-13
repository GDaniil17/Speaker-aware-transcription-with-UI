"""OpenAI Whisper via MLX (Apple Silicon GPU) or faster-whisper (CUDA/CPU)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

from speaker_transcribe.audio import read_wav_samples
from speaker_transcribe.catalog import mlx_repo_for
from speaker_transcribe.config import Settings
from speaker_transcribe.engines.common import ProgressBarFn, ProgressFn, cached_model, log
from speaker_transcribe.hub import ensure_whisper_model_downloaded
from speaker_transcribe.transcript import TranscriptSegment, speaker_for


def transcribe_mlx(
    wav_path: Path,
    settings: Settings,
    diar: list[dict],
    sp_map: dict,
    progress: ProgressFn | None,
    on_progress: ProgressBarFn | None = None,
) -> Iterator[TranscriptSegment]:
    """Transcribe with Apple's MLX (single blocking call, no streaming progress)."""
    import mlx_whisper

    repo = mlx_repo_for(settings.whisper_model)
    log(progress, f"Loading Whisper model ({settings.whisper_model})…")
    ensure_whisper_model_downloaded(settings.whisper_model, settings.backend, on_progress)
    log(progress, f"Transcribing with MLX ({repo})…")
    if on_progress:
        on_progress("Transcribing (MLX, no live progress)", 0.0, None)
    result = mlx_whisper.transcribe(
        # Samples rather than a path: mlx-whisper would decode a path by running
        # a binary named `ffmpeg`, which a machine without one lacks.
        read_wav_samples(wav_path),
        path_or_hf_repo=repo,
        language=settings.language,
        temperature=settings.temperature,
        compression_ratio_threshold=settings.compression_ratio_threshold,
        logprob_threshold=settings.log_prob_threshold,
        no_speech_threshold=settings.no_speech_threshold,
        condition_on_previous_text=settings.condition_on_previous_text,
        initial_prompt=settings.initial_prompt,
        hallucination_silence_threshold=settings.hallucination_silence_threshold,
        word_timestamps=False,
        verbose=None,
    )

    segments = result.get("segments") or []
    total = segments[-1]["end"] if segments else (settings.duration or None)
    for seg in segments:
        if on_progress and total:
            on_progress("Transcribing", min(seg["end"] / total, 1.0), 0.0)
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        speaker = speaker_for(seg["start"], seg["end"], diar, sp_map)
        yield TranscriptSegment(seg["start"], seg["end"], speaker, text)

    if on_progress:
        on_progress("Transcribing", 1.0, 0.0)


def transcribe_faster_whisper(
    wav_path: Path,
    settings: Settings,
    diar: list[dict],
    sp_map: dict,
    progress: ProgressFn | None,
    on_progress: ProgressBarFn | None = None,
) -> Iterator[TranscriptSegment]:
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    log(progress, f"Loading Whisper model ({settings.whisper_model})…")
    ensure_whisper_model_downloaded(settings.whisper_model, settings.backend, on_progress)
    pipeline = cached_model(
        ("faster-whisper", settings.whisper_model, settings.device, settings.compute_type),
        lambda: BatchedInferencePipeline(
            WhisperModel(
                settings.whisper_model,
                device=settings.device,
                compute_type=settings.compute_type,
            )
        ),
    )
    pipeline.chunk_length = settings.chunk_length

    log(progress, "Transcribing…")
    if on_progress:
        on_progress("Transcribing", 0.0, None)
    segments, info = pipeline.transcribe(
        str(wav_path),
        language=settings.language,
        beam_size=settings.beam_size,
        temperature=settings.temperature,
        compression_ratio_threshold=settings.compression_ratio_threshold,
        log_prob_threshold=settings.log_prob_threshold,
        no_speech_threshold=settings.no_speech_threshold,
        condition_on_previous_text=settings.condition_on_previous_text,
        initial_prompt=settings.initial_prompt,
        repetition_penalty=settings.repetition_penalty,
        no_repeat_ngram_size=settings.no_repeat_ngram_size,
        word_timestamps=False,
        vad_filter=settings.vad_filter,
        hallucination_silence_threshold=settings.hallucination_silence_threshold,
        batch_size=settings.batch_size,
        chunk_length=settings.chunk_length,
    )

    total = getattr(info, "duration", None) or settings.duration
    started = time.monotonic()

    for seg in segments:
        if on_progress and total:
            frac = min(seg.end / total, 1.0)
            elapsed = time.monotonic() - started
            eta = (elapsed / frac - elapsed) if frac > 0 else None
            on_progress("Transcribing", frac, eta)

        text = seg.text.strip()
        if not text:
            continue
        speaker = speaker_for(seg.start, seg.end, diar, sp_map)
        yield TranscriptSegment(seg.start, seg.end, speaker, text)

    if on_progress:
        on_progress("Transcribing", 1.0, 0.0)
