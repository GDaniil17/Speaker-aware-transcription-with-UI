"""Runtime configuration and device detection."""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

WHISPER_MODELS: dict[str, str] = {
    "tiny": "multilingual · fastest, lowest quality",
    "base": "multilingual · very fast",
    "small": "multilingual · fast, good for testing",
    "medium": "multilingual · balanced",
    "large-v3": "multilingual · best quality (slow on CPU)",
    "large-v3-turbo": "multilingual · near-large quality, much faster",
    "turbo": "multilingual · alias of large-v3-turbo",
    "distil-large-v3": "English only · fast distilled large",
    "distil-large-v3.5": "English only · newest distilled large",
    "distil-medium.en": "English only · fast distilled medium",
    "distil-small.en": "English only · fastest distilled",
}

MLX_MODEL_REPOS: dict[str, str] = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "distil-large-v3": "mlx-community/distil-whisper-large-v3",
    "distil-large-v3.5": "mlx-community/distil-whisper-large-v3.5",
    "distil-medium.en": "mlx-community/distil-whisper-medium.en",
    "distil-small.en": "mlx-community/distil-whisper-small.en",
}


WHISPER_LANGUAGES: dict[str, str] = {
    "en": "english", "zh": "chinese", "de": "german", "es": "spanish", "ru": "russian",
    "ko": "korean", "fr": "french", "ja": "japanese", "pt": "portuguese", "tr": "turkish",
    "pl": "polish", "ca": "catalan", "nl": "dutch", "ar": "arabic", "sv": "swedish",
    "it": "italian", "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese",
    "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay", "cs": "czech",
    "ro": "romanian", "da": "danish", "hu": "hungarian", "ta": "tamil", "no": "norwegian",
    "th": "thai", "ur": "urdu", "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian",
    "la": "latin", "mi": "maori", "ml": "malayalam", "cy": "welsh", "sk": "slovak",
    "te": "telugu", "fa": "persian", "lv": "latvian", "bn": "bengali", "sr": "serbian",
    "az": "azerbaijani", "sl": "slovenian", "kn": "kannada", "et": "estonian",
    "mk": "macedonian", "br": "breton", "eu": "basque", "is": "icelandic", "hy": "armenian",
    "ne": "nepali", "mn": "mongolian", "bs": "bosnian", "kk": "kazakh", "sq": "albanian",
    "sw": "swahili", "gl": "galician", "mr": "marathi", "pa": "punjabi", "si": "sinhala",
    "km": "khmer", "sn": "shona", "yo": "yoruba", "so": "somali", "af": "afrikaans",
    "oc": "occitan", "ka": "georgian", "be": "belarusian", "tg": "tajik", "sd": "sindhi",
    "gu": "gujarati", "am": "amharic", "yi": "yiddish", "lo": "lao", "uz": "uzbek",
    "fo": "faroese", "ht": "haitian creole", "ps": "pashto", "tk": "turkmen",
    "nn": "nynorsk", "mt": "maltese", "sa": "sanskrit", "lb": "luxembourgish",
    "my": "myanmar", "bo": "tibetan", "tl": "tagalog", "mg": "malagasy", "as": "assamese",
    "tt": "tatar", "haw": "hawaiian", "ln": "lingala", "ha": "hausa", "ba": "bashkir",
    "jw": "javanese", "su": "sundanese", "yue": "cantonese",
}


def is_english_only_model(model: str) -> bool:
    """True for Whisper variants that only transcribe English (distil-* and *.en)."""
    return "distil" in model or model.endswith(".en")


def mlx_repo_for(model: str) -> str:
    """HF repo for an MLX Whisper model, with a fallback for unknown keys."""
    return MLX_MODEL_REPOS.get(model, f"mlx-community/whisper-{model}-mlx")


def whisper_repo_for(model: str, backend: str) -> str | None:
    """HuggingFace repo a (resolved) backend maps a model to, or None if unknown."""
    if backend == "mlx":
        return mlx_repo_for(model)
    try:
        from faster_whisper.utils import _MODELS

        return _MODELS.get(model)
    except Exception:
        return None


def downloaded_whisper_models(backend: str) -> set[str]:
    """WHISPER_MODELS keys already in the HuggingFace cache for `backend`."""
    try:
        from huggingface_hub import scan_cache_dir

        cached = {repo.repo_id for repo in scan_cache_dir().repos}
    except Exception:
        return set()

    ready: set[str] = set()
    for model in WHISPER_MODELS:
        repo = whisper_repo_for(model, backend)
        if repo and repo in cached:
            ready.add(model)
    return ready


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def mlx_available() -> bool:
    """True when the mlx-whisper backend can be used (installed, Apple Silicon)."""
    return is_apple_silicon() and importlib.util.find_spec("mlx_whisper") is not None


def detect_backend() -> str:
    """`mlx` on Apple Silicon when available, otherwise `faster-whisper`."""
    return "mlx" if mlx_available() else "faster-whisper"


def describe_runtime(backend: str, device: str, compute_type: str) -> str:
    """Short human label of where transcription will run."""
    if backend == "mlx":
        return "MLX (Apple Silicon GPU)"
    return f"faster-whisper · {device} ({compute_type})"


def detect_device() -> tuple[str, str]:
    """Best available (device, compute_type) for faster-whisper: cuda or cpu/int8."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def detect_diarization_device() -> str:
    """Best torch device for pyannote: cuda > mps (Apple GPU) > cpu."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def auto_beam_size(device: str) -> int:
    """Beam size on auto: 1 (greedy) on CPU, 5 on GPU."""
    return 5 if device == "cuda" else 1


def get_hf_token() -> str | None:
    """HuggingFace token from the environment (HF_TOKEN or HUGGINGFACE_TOKEN)."""
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or None


@dataclass
class Settings:
    """All knobs for a single transcription run."""

    input_path: str = ""
    start: float = 0.0
    duration: float | None = None

    whisper_model: str = "large-v3"
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    language: str | None = "ru"

    diarize: bool = True
    min_speakers: int = 2
    max_speakers: int = 5

    backend: str = "auto"
    beam_size: int | None = None
    vad_filter: bool = True

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
