"""Local speaker-aware transcription (Whisper + pyannote diarization)."""

from speaker_transcribe.config import Settings
from speaker_transcribe.pipeline import TranscriptSegment, run_pipeline

__all__ = ["Settings", "TranscriptSegment", "run_pipeline"]
__version__ = "0.1.0"
