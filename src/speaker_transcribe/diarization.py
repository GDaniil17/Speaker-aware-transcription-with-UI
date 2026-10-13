"""Speaker diarization with pyannote.audio."""

from __future__ import annotations

from pathlib import Path

from speaker_transcribe.config import Settings
from speaker_transcribe.engines.common import ProgressBarFn, ProgressFn, log
from speaker_transcribe.hub import local_repo_snapshot


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


def diarize(
    wav_path: Path,
    settings: Settings,
    progress: ProgressFn | None,
    on_progress: ProgressBarFn | None = None,
):
    """Run pyannote diarization, returning (segments, speaker_map)."""
    import torch
    from pyannote.audio import Pipeline as DiarizationPipeline

    # A downloaded pipeline loads from disk: no token and no network needed.
    # The token is only for the first download of the gated model.
    local = local_repo_snapshot(settings.diarization_model)
    pipeline = None
    if local is not None:
        log(progress, "Loading diarization model (local copy)…")
        try:
            pipeline = DiarizationPipeline.from_pretrained(local)
        except Exception as exc:
            if not settings.hf_token:
                raise
            log(progress, f"Local diarization model incomplete ({exc}); downloading…")
    if pipeline is None:
        if not settings.hf_token:
            raise RuntimeError(
                "A HuggingFace token is required to download the diarization model "
                "once. Set HF_TOKEN in your environment/.env and accept the model "
                f"terms at https://huggingface.co/{settings.diarization_model}"
            )
        log(progress, "Downloading diarization model…")
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
    log(progress, f"Diarizing on {device} (detecting speakers)…")
    try:
        output = run_on(device)
    except Exception as exc:
        if device == "cpu":
            raise
        log(progress, f"{device} diarization failed ({exc}); retrying on CPU…")
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
    log(progress, f"Diarization done — {len(unique)} speaker(s) found.")
    return segments, speaker_map
