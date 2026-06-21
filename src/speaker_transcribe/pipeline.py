"""Core pipeline: speaker diarization (pyannote) + transcription (Whisper)."""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from speaker_transcribe.audio import prepare_audio
from speaker_transcribe.config import Settings, mlx_repo_for

ProgressFn = Callable[[str], None]
ProgressBarFn = Callable[[str, float, float | None], None]


@dataclass
class TranscriptSegment:
    """One transcribed chunk attributed to a speaker."""

    start: float
    end: float
    speaker: str
    text: str


def _log(progress: ProgressFn | None, msg: str) -> None:
    if progress:
        progress(msg)


def _make_diarization_hook(progress: ProgressFn | None, on_progress: ProgressBarFn | None):
    """Adapt pyannote's hook protocol to our progress callbacks."""
    if not progress and not on_progress:
        return None

    last = {"step": None}

    def hook(step_name, step_artifact=None, file=None, completed=None, total=None):
        if completed is not None and total:
            frac = completed / max(total, 1)
            if on_progress:
                on_progress(f"Diarizing · {step_name}", frac, None)
            if progress and step_name != last["step"]:
                progress(f"Diarizing: {step_name}…")
                last["step"] = step_name

    return hook


def _diarize(
    wav_path: Path,
    settings: Settings,
    progress: ProgressFn | None,
    on_progress: ProgressBarFn | None = None,
):
    """Run pyannote diarization, returning (segments, speaker_map)."""
    import torch
    from pyannote.audio import Pipeline as DiarizationPipeline

    if not settings.hf_token:
        raise RuntimeError(
            "A HuggingFace token is required for diarization. Set HF_TOKEN "
            "in your environment/.env and accept the model terms at "
            f"https://huggingface.co/{settings.diarization_model}"
        )

    _log(progress, "Loading diarization model…")
    pipeline = DiarizationPipeline.from_pretrained(
        settings.diarization_model, token=settings.hf_token
    )
    hook = _make_diarization_hook(progress, on_progress)

    def run_on(device: str):
        pipeline.to(torch.device(device))
        return pipeline(
            str(wav_path),
            min_speakers=settings.min_speakers,
            max_speakers=settings.max_speakers,
            hook=hook,
        )

    device = settings.diarization_device or "cpu"
    _log(progress, f"Diarizing on {device} (detecting speakers)…")
    try:
        output = run_on(device)
    except Exception as exc:
        if device == "cpu":
            raise
        _log(progress, f"{device} diarization failed ({exc}); retrying on CPU…")
        output = run_on("cpu")

    segments: list[dict] = []
    if hasattr(output, "speaker_diarization"):
        for turn, speaker in output.speaker_diarization:
            segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    elif hasattr(output, "itertracks"):
        for turn, _, speaker in output.itertracks(yield_label=True):
            segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    else:
        raise RuntimeError("Unknown pyannote output format.")

    unique = sorted({s["speaker"] for s in segments})
    speaker_map = {spk: f"Speaker {i + 1}" for i, spk in enumerate(unique)}
    _log(progress, f"Diarization done — {len(unique)} speaker(s) found.")
    return segments, speaker_map


def _speaker_for(seg_start: float, seg_end: float, diar: list[dict], sp_map: dict) -> str:
    """Attribute a transcription segment to the speaker with the most overlap."""
    best_speaker = "Unknown"
    best_overlap = 0.0
    for ds in diar:
        overlap = max(0.0, min(seg_end, ds["end"]) - max(seg_start, ds["start"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = sp_map[ds["speaker"]]
    return best_speaker


def _transcribe(
    wav_path: Path,
    settings: Settings,
    diar: list[dict],
    sp_map: dict,
    progress: ProgressFn | None,
    on_progress: ProgressBarFn | None = None,
) -> Iterator[TranscriptSegment]:
    """Dispatch to the configured transcription backend."""
    if settings.backend == "mlx":
        yield from _transcribe_mlx(wav_path, settings, diar, sp_map, progress, on_progress)
    else:
        yield from _transcribe_faster_whisper(
            wav_path, settings, diar, sp_map, progress, on_progress
        )


def _transcribe_mlx(
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
    _log(progress, f"Transcribing with MLX ({repo})…")
    if on_progress:
        on_progress("Transcribing (MLX, no live progress)", 0.0, None)
    result = mlx_whisper.transcribe(
        str(wav_path),
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
        speaker = _speaker_for(seg["start"], seg["end"], diar, sp_map)
        yield TranscriptSegment(seg["start"], seg["end"], speaker, text)

    if on_progress:
        on_progress("Transcribing", 1.0, 0.0)


def _transcribe_faster_whisper(
    wav_path: Path,
    settings: Settings,
    diar: list[dict],
    sp_map: dict,
    progress: ProgressFn | None,
    on_progress: ProgressBarFn | None = None,
) -> Iterator[TranscriptSegment]:
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    _log(progress, f"Loading Whisper model ({settings.whisper_model})…")
    model = WhisperModel(
        settings.whisper_model,
        device=settings.device,
        compute_type=settings.compute_type,
    )
    pipeline = BatchedInferencePipeline(model)
    pipeline.chunk_length = settings.chunk_length

    _log(progress, "Transcribing…")
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
        speaker = _speaker_for(seg.start, seg.end, diar, sp_map)
        yield TranscriptSegment(seg.start, seg.end, speaker, text)

    if on_progress:
        on_progress("Transcribing", 1.0, 0.0)


def run_pipeline(
    settings: Settings,
    progress: ProgressFn | None = None,
    on_progress: ProgressBarFn | None = None,
) -> list[TranscriptSegment]:
    """End-to-end: prepare audio, diarize, transcribe, attribute speakers."""
    with tempfile.TemporaryDirectory(prefix="spk_") as tmp:
        wav_path = Path(tmp) / "input.wav"
        _log(progress, "Preparing audio…")
        prepare_audio(
            settings.input_path,
            wav_path,
            start=settings.start,
            duration=settings.duration,
        )

        if settings.diarize:
            diar, sp_map = _diarize(wav_path, settings, progress, on_progress)
        else:
            _log(progress, "Skipping diarization — transcription only.")
            diar, sp_map = [], {}
        result = list(_transcribe(wav_path, settings, diar, sp_map, progress, on_progress))

        if not settings.diarize:
            for seg in result:
                seg.speaker = "Speaker 1"

    if settings.start:
        for seg in result:
            seg.start += settings.start
            seg.end += settings.start

    _log(progress, f"Done — {len(result)} transcript segment(s).")
    return result
