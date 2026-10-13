"""Local speaker-aware transcription: Whisper and other ASR engines + pyannote."""

from speaker_transcribe.config import Settings
from speaker_transcribe.pipeline import run_pipeline
from speaker_transcribe.transcript import TranscriptSegment

__all__ = ["Settings", "TranscriptSegment", "run_pipeline"]
__version__ = "0.1.0"
