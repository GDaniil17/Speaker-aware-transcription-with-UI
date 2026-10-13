"""Settings for one transcription run, and environment defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from speaker_transcribe.catalog import DEFAULT_WHISPER_MODEL, DIARIZATION_MODEL
from speaker_transcribe.chunking import DEFAULT_CHUNK_LADDER, DEFAULT_STALL_TIMEOUT
from speaker_transcribe.runtime import (
    auto_beam_size,
    detect_backend,
    detect_device,
    detect_diarization_device,
)

load_dotenv()

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# pyannote.audio 4 sends usage metrics to otel.pyannote.ai unless told not to.
# Everything here runs locally, so it is off unless the user opts back in.
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")


def get_hf_token() -> str | None:
    """HuggingFace token from the environment (HF_TOKEN or HUGGINGFACE_TOKEN)."""
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or None


@dataclass
class Settings:
    """All knobs for a single transcription run."""

    input_path: str = ""
    start: float = 0.0
    duration: float | None = None

    engine: str = "whisper"
    whisper_model: str = DEFAULT_WHISPER_MODEL
    diarization_model: str = DIARIZATION_MODEL
    language: str | None = None  # None = auto-detect

    diarize: bool = True
    min_speakers: int = 2
    max_speakers: int = 5

    backend: str = "auto"
    beam_size: int | None = None
    vad_filter: bool = True
    temperature: float | tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    condition_on_previous_text: bool = True
    initial_prompt: str | None = None
    compression_ratio_threshold: float | None = 2.4
    log_prob_threshold: float | None = -1.0
    no_speech_threshold: float | None = 0.6
    hallucination_silence_threshold: float | None = None
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    batch_size: int = 8
    chunk_length: int = 60

    # Stall recovery. A run that makes no progress for `stall_timeout` seconds is
    # given up on and retried in chunks of `chunk_ladder[0]` seconds, then
    # `chunk_ladder[1]`, and so on. Set `stall_timeout` to 0 to never interrupt a
    # run (transcription then also stays in this process, as it used to).
    stall_timeout: float = DEFAULT_STALL_TIMEOUT
    chunk_fallback: bool = True
    chunk_ladder: tuple[float, ...] = DEFAULT_CHUNK_LADDER
    try_whole_file_first: bool = True
    skip_failed_chunks: bool = True

    device: str = ""
    compute_type: str = ""
    diarization_device: str = ""
    hf_token: str | None = field(default_factory=get_hf_token)

    def __post_init__(self) -> None:
        if not self.device or not self.compute_type:
            dev, ct = detect_device()
            self.device = self.device or dev
            self.compute_type = self.compute_type or ct
        if not self.diarization_device:
            self.diarization_device = detect_diarization_device()
        if self.backend == "auto":
            self.backend = detect_backend()
        if self.beam_size is None:
            self.beam_size = auto_beam_size(self.device)
