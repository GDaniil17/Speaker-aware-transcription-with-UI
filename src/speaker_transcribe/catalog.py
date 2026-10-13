"""What can be selected: engines, models, languages and where they download from.

Pure data plus a few lookups. Nothing here imports a model library, so the UI,
the CLI and the tests can read the catalog cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

DEFAULT_WHISPER_MODEL = "large-v3-turbo"

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
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}


# --- Non-Whisper engines -------------------------------------------------------
#
# Each engine produces per-segment timestamps, which is what speaker attribution
# (transcript.speaker_for) needs. Their heavy deps are optional extras, imported
# lazily by the modules in `speaker_transcribe.engines`.

# Parakeet auto-detects its language; this list is informational (selector is
# disabled in the UI). 25 European languages, per the nvidia/parakeet-tdt-0.6b-v3
# model card.
PARAKEET_LANGUAGES: dict[str, str] = {
    "bg": "bulgarian",
    "hr": "croatian",
    "cs": "czech",
    "da": "danish",
    "nl": "dutch",
    "en": "english",
    "et": "estonian",
    "fi": "finnish",
    "fr": "french",
    "de": "german",
    "el": "greek",
    "hu": "hungarian",
    "it": "italian",
    "lv": "latvian",
    "lt": "lithuanian",
    "mt": "maltese",
    "pl": "polish",
    "pt": "portuguese",
    "ro": "romanian",
    "sk": "slovak",
    "sl": "slovenian",
    "es": "spanish",
    "sv": "swedish",
    "ru": "russian",
    "uk": "ukrainian",
}

# Qwen3-ASR takes a language *name* (e.g. "English"), not a code, so we map
# code -> the name the model expects. 30 languages, per the Qwen3-ASR-1.7B card.
QWEN3_ASR_LANGUAGES: dict[str, str] = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "hu": "Hungarian",
    "mk": "Macedonian",
    "ro": "Romanian",
}

# SenseVoiceSmall recognises 5 languages and emits CTC-alignment timestamps.
SENSEVOICE_LANGUAGES: dict[str, str] = {
    "zh": "Chinese",
    "yue": "Cantonese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
}

GIGAAM_LANGUAGES: dict[str, str] = {"ru": "Russian"}

PARAKEET_MODEL = "parakeet-tdt-0.6b-v3"
QWEN3_ASR_MODEL = "Qwen3-ASR-1.7B"
SENSEVOICE_MODEL = "SenseVoiceSmall"

# repo ids the engines download from HuggingFace
PARAKEET_REPOS = {
    "mlx": "mlx-community/parakeet-tdt-0.6b-v3",
    "nemo": "nvidia/parakeet-tdt-0.6b-v3",
}
QWEN3_ASR_REPO = "Qwen/Qwen3-ASR-1.7B"
QWEN3_ALIGNER_REPO = "Qwen/Qwen3-ForcedAligner-0.6B"
SENSEVOICE_REPO = "FunAudioLLM/SenseVoiceSmall"
SENSEVOICE_VAD_REPO = "funasr/fsmn-vad"

# GigaAM (Sber) exported to ONNX for onnx-asr, plus Silero VAD that cuts long
# audio into the <=20 s pieces the model accepts. int8 weights: a quarter of the
# size, same text in practice. The v3 "e2e" models add punctuation, capitals and
# number normalisation; the others write lowercase words only. The first model
# is the default.
GIGAAM_VERSIONS: dict[str, str] = {
    "v3": "GigaAM v3",
    "v2": "GigaAM v2",
}
GIGAAM_MODELS: dict[str, str] = {
    "gigaam-v3-e2e-rnnt": "Russian · punctuation and capitals, RNN-T decoder, best accuracy",
    "gigaam-v3-e2e-ctc": "Russian · punctuation and capitals, CTC decoder, slightly faster",
    "gigaam-v3-rnnt": "Russian · lowercase, no punctuation, RNN-T decoder",
    "gigaam-v3-ctc": "Russian · lowercase, no punctuation, CTC decoder",
    "gigaam-v2-rnnt": "Russian · lowercase, no punctuation, RNN-T decoder",
    "gigaam-v2-ctc": "Russian · lowercase, no punctuation, CTC decoder",
}
GIGAAM_REPOS: dict[str, str] = {
    "v3": "istupakov/gigaam-v3-onnx",
    "v2": "istupakov/gigaam-v2-onnx",
}
GIGAAM_VAD_REPO = "istupakov/silero-vad-onnx"
GIGAAM_QUANTIZATION = "int8"
# Files each model needs from its repo (fnmatch patterns, as onnx-asr names them).
GIGAAM_FILES: dict[str, list[str]] = {
    "gigaam-v3-e2e-rnnt": ["config.json", "v3_e2e_rnnt_vocab.txt", "v3_e2e_rnnt_*.int8.onnx"],
    "gigaam-v3-e2e-ctc": ["config.json", "v3_e2e_ctc_vocab.txt", "v3_e2e_ctc.int8.onnx"],
    "gigaam-v3-rnnt": ["config.json", "v3_vocab.txt", "v3_rnnt_*.int8.onnx"],
    "gigaam-v3-ctc": ["config.json", "v3_vocab.txt", "v3_ctc.int8.onnx"],
    "gigaam-v2-rnnt": ["config.json", "v2_vocab.txt", "v2_rnnt_*.int8.onnx"],
    "gigaam-v2-ctc": ["config.json", "v2_vocab.txt", "v2_ctc.int8.onnx"],
}
GIGAAM_VAD_FILES = ["config.json", "silero_vad.onnx"]


@dataclass(frozen=True)
class Engine:
    """A selectable ASR engine (Whisper or an alternative model)."""

    key: str
    label: str
    description: str
    models: dict[str, str]  # model id -> short description
    languages: dict[str, str] | None  # code -> name, or None when auto-only
    install_extra: str  # pip extra to install, "" for Whisper (always available)


ENGINES: dict[str, Engine] = {
    "whisper": Engine(
        key="whisper",
        label="Whisper",
        description="OpenAI Whisper · multilingual, MLX or faster-whisper backends.",
        models=WHISPER_MODELS,
        languages=WHISPER_LANGUAGES,
        install_extra="",
    ),
    "parakeet": Engine(
        key="parakeet",
        label="Parakeet (NVIDIA)",
        description="parakeet-tdt-0.6b-v3 · 25 European languages, auto-detected.",
        models={PARAKEET_MODEL: "25 European languages · fast, auto language detect"},
        languages=None,
        # resolved per platform in engine_install.engine_install_extra
        install_extra="parakeet",
    ),
    "qwen3-asr": Engine(
        key="qwen3-asr",
        label="Qwen3-ASR",
        description="Qwen3-ASR-1.7B · 30 languages, timestamps via forced aligner.",
        models={QWEN3_ASR_MODEL: "30 languages · LLM-based ASR with forced aligner"},
        languages=QWEN3_ASR_LANGUAGES,
        install_extra="qwen3-asr",
    ),
    "sensevoice": Engine(
        key="sensevoice",
        label="SenseVoice",
        description="SenseVoiceSmall · 5 languages (zh/yue/en/ja/ko), fast, CTC timestamps.",
        models={SENSEVOICE_MODEL: "5 languages · 15x faster than Whisper, CTC timestamps"},
        languages=SENSEVOICE_LANGUAGES,
        install_extra="sensevoice",
    ),
    "gigaam": Engine(
        key="gigaam",
        label="GigaAM (Sber)",
        description="GigaAM v3 / v2 · Russian only, fast on CPU (ONNX); "
        "v3 e2e adds punctuation and capitals.",
        models=GIGAAM_MODELS,
        languages=GIGAAM_LANGUAGES,
        install_extra="",  # onnx-asr is a core dependency
    ),
}


# --- Hardware-aware model recommendations -------------------------------------
#
# Shown by the Streamlit UI ("Which model should I choose?"); the README repeats
# it. Keyed by the acceleration profile returned by `runtime.hardware_profile()`.

HARDWARE_LABELS: dict[str, str] = {
    "cuda": "NVIDIA GPU (CUDA)",
    "mlx": "Apple Silicon (MLX / MPS)",
    "cpu": "CPU only",
}

# Full code -> name map per engine, for the "Supported languages" UI list.
ENGINE_LANGUAGE_LISTS: dict[str, dict[str, str]] = {
    "whisper": WHISPER_LANGUAGES,
    "parakeet": PARAKEET_LANGUAGES,
    "qwen3-asr": QWEN3_ASR_LANGUAGES,
    "sensevoice": SENSEVOICE_LANGUAGES,
    "gigaam": GIGAAM_LANGUAGES,
}

# Per-hardware picks. Each entry: (engine label, recommended model, why).
MODEL_GUIDANCE: dict[str, list[tuple[str, str, str]]] = {
    "cuda": [
        (
            "Whisper",
            "large-v3 / large-v3-turbo",
            "Best multilingual quality; float16 on CUDA keeps large models fast.",
        ),
        (
            "Parakeet",
            "parakeet-tdt-0.6b-v3",
            "Very fast for the 25 EU languages, language auto-detected.",
        ),
        (
            "Qwen3-ASR",
            "Qwen3-ASR-1.7B",
            "Strong accuracy across 30 languages; the heaviest, wants GPU VRAM.",
        ),
    ],
    "mlx": [
        ("Whisper (MLX)", "large-v3-turbo", "Near-large quality at high speed on the Apple GPU."),
        (
            "Parakeet (MLX)",
            "parakeet-tdt-0.6b-v3",
            "Fast EU-language transcription via parakeet-mlx (avoids heavy NeMo).",
        ),
        (
            "SenseVoice",
            "SenseVoiceSmall",
            "Good for Chinese, Cantonese, English, Japanese, and Korean audio.",
        ),
        (
            "GigaAM v3",
            "gigaam-v3-e2e-rnnt",
            "Russian audio with punctuation, very fast on the CPU cores.",
        ),
    ],
    "cpu": [
        (
            "Whisper",
            "small / medium",
            "Best speed/quality balance on CPU (int8); avoid large-v3 (slow).",
        ),
        (
            "Whisper distil-*",
            "distil-small.en / distil-medium.en",
            "Fastest option for English-only audio.",
        ),
        (
            "SenseVoice",
            "SenseVoiceSmall",
            "~15× faster than Whisper for its 5 languages — great on CPU.",
        ),
        (
            "GigaAM v3",
            "gigaam-v3-e2e-rnnt",
            "Best choice for Russian on CPU: fast ONNX model, no PyTorch needed.",
        ),
    ],
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


def gigaam_version(model: str) -> str:
    """'v3' for gigaam-v3-e2e-rnnt and the like."""
    return model.split("-")[1]


def gigaam_repo_for(model: str) -> str:
    """Hugging Face repo holding a GigaAM model's ONNX files."""
    return GIGAAM_REPOS[gigaam_version(model)]
