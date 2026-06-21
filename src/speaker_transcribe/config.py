"""Runtime configuration and device detection."""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
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


# --- Non-Whisper engines -------------------------------------------------------
#
# Each engine produces per-segment timestamps, which is what speaker attribution
# (pipeline._speaker_for) needs. Their heavy deps are optional extras, imported
# lazily in pipeline.py and probed below.

# Parakeet auto-detects its language; this list is informational (selector is
# disabled in the UI). 25 European languages, per the nvidia/parakeet-tdt-0.6b-v3
# model card.
PARAKEET_LANGUAGES: dict[str, str] = {
    "bg": "bulgarian", "hr": "croatian", "cs": "czech", "da": "danish",
    "nl": "dutch", "en": "english", "et": "estonian", "fi": "finnish",
    "fr": "french", "de": "german", "el": "greek", "hu": "hungarian",
    "it": "italian", "lv": "latvian", "lt": "lithuanian", "mt": "maltese",
    "pl": "polish", "pt": "portuguese", "ro": "romanian", "sk": "slovak",
    "sl": "slovenian", "es": "spanish", "sv": "swedish", "ru": "russian",
    "uk": "ukrainian",
}

# Qwen3-ASR takes a language *name* (e.g. "English"), not a code, so we map
# code -> the name the model expects. 30 languages, per the Qwen3-ASR-1.7B card.
QWEN3_ASR_LANGUAGES: dict[str, str] = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic",
    "de": "German", "fr": "French", "es": "Spanish", "pt": "Portuguese",
    "id": "Indonesian", "it": "Italian", "ko": "Korean", "ru": "Russian",
    "th": "Thai", "vi": "Vietnamese", "ja": "Japanese", "tr": "Turkish",
    "hi": "Hindi", "ms": "Malay", "nl": "Dutch", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech",
    "fil": "Filipino", "fa": "Persian", "el": "Greek", "hu": "Hungarian",
    "mk": "Macedonian", "ro": "Romanian",
}

# SenseVoiceSmall recognises 5 languages and emits CTC-alignment timestamps.
SENSEVOICE_LANGUAGES: dict[str, str] = {
    "zh": "Chinese", "yue": "Cantonese", "en": "English", "ja": "Japanese",
    "ko": "Korean",
}

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
        # resolved per-platform in engine_install_hint (parakeet-mlx on Apple Silicon)
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
}


# --- Hardware-aware model recommendations -------------------------------------
#
# Shared by the Streamlit UI ("Which engine & model should I use?") and the
# README so both stay in sync. Keyed by the acceleration profile returned by
# `hardware_profile()`.

HARDWARE_LABELS: dict[str, str] = {
    "cuda": "NVIDIA GPU (CUDA)",
    "mlx": "Apple Silicon (MLX / MPS)",
    "cpu": "CPU only",
}

# One-line language coverage per engine, for the comparison table.
ENGINE_LANGUAGE_SUMMARY: dict[str, str] = {
    "whisper": "99 languages (multilingual); distil-* variants are English-only",
    "parakeet": "25 European languages, auto-detected",
    "qwen3-asr": "30 languages",
    "sensevoice": "5 languages: Chinese, Cantonese, English, Japanese, Korean",
}

# Full code -> name map per engine, for the "Supported languages" UI list.
ENGINE_LANGUAGE_LISTS: dict[str, dict[str, str]] = {
    "whisper": WHISPER_LANGUAGES,
    "parakeet": PARAKEET_LANGUAGES,
    "qwen3-asr": QWEN3_ASR_LANGUAGES,
    "sensevoice": SENSEVOICE_LANGUAGES,
}

# Per-hardware picks. Each entry: (engine label, recommended model, why).
MODEL_GUIDANCE: dict[str, list[tuple[str, str, str]]] = {
    "cuda": [
        ("Whisper", "large-v3 / large-v3-turbo",
         "Best multilingual quality; float16 on CUDA keeps large models fast."),
        ("Parakeet", "parakeet-tdt-0.6b-v3",
         "Very fast for the 25 EU languages, language auto-detected."),
        ("Qwen3-ASR", "Qwen3-ASR-1.7B",
         "Strong accuracy across 30 languages; the heaviest, wants GPU VRAM."),
    ],
    "mlx": [
        ("Whisper (MLX)", "large-v3-turbo",
         "Near-large quality at high speed on the Apple GPU."),
        ("Parakeet (MLX)", "parakeet-tdt-0.6b-v3",
         "Fast EU-language transcription via parakeet-mlx (avoids heavy NeMo)."),
        ("SenseVoice", "SenseVoiceSmall",
         "Best pick when audio is Chinese/Cantonese/English/Japanese/Korean."),
    ],
    "cpu": [
        ("Whisper", "small / medium",
         "Best speed/quality balance on CPU (int8); avoid large-v3 (slow)."),
        ("Whisper distil-*", "distil-small.en / distil-medium.en",
         "Fastest option for English-only audio."),
        ("SenseVoice", "SenseVoiceSmall",
         "~15× faster than Whisper for its 5 languages — great on CPU."),
    ],
}


def hardware_profile() -> str:
    """Detected acceleration profile for recommendations: 'cuda', 'mlx', or 'cpu'."""
    if detect_device()[0] == "cuda":
        return "cuda"
    if is_apple_silicon():
        return "mlx"
    return "cpu"


def parakeet_available() -> bool:
    """True when a Parakeet backend is importable (MLX on Apple Silicon, else NeMo)."""
    if is_apple_silicon() and importlib.util.find_spec("parakeet_mlx") is not None:
        return True
    return importlib.util.find_spec("nemo") is not None


def qwen3_asr_available() -> bool:
    """True when the qwen-asr package is importable."""
    return importlib.util.find_spec("qwen_asr") is not None


def sensevoice_available() -> bool:
    """True when the funasr package (SenseVoice backend) is importable."""
    return importlib.util.find_spec("funasr") is not None


def engine_available(engine: str) -> bool:
    """Whether the packages for `engine` are installed."""
    if engine == "whisper":
        return True
    if engine == "parakeet":
        return parakeet_available()
    if engine == "qwen3-asr":
        return qwen3_asr_available()
    if engine == "sensevoice":
        return sensevoice_available()
    return False


def engine_install_hint(engine: str) -> str:
    """The `make` command that installs `engine`'s optional dependencies."""
    extra = ENGINES[engine].install_extra
    if engine == "parakeet" and is_apple_silicon():
        extra = "parakeet-mlx"
    return f"make install-{extra}" if extra else ""


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


def _repo_in_cache(repo_id: str) -> bool:
    """True when `repo_id` is already present in the local HuggingFace cache."""
    try:
        from huggingface_hub import scan_cache_dir

        return repo_id in {repo.repo_id for repo in scan_cache_dir().repos}
    except Exception:
        return False


def ensure_whisper_model_downloaded(
    model: str,
    backend: str,
    on_progress: Callable[[str, float, float | None], None] | None = None,
) -> None:
    """Download a Whisper model snapshot when it is missing from the local HF cache."""
    repo_id = whisper_repo_for(model, backend)
    if repo_id:
        ensure_repo_downloaded(repo_id, on_progress)


def _human_bytes(num: float) -> str:
    """Format a byte count as a compact human-readable string (e.g. '1.4 GB')."""
    for unit in ("B", "KB", "MB"):
        if num < 1024:
            return f"{num:.0f} {unit}"
        num /= 1024
    return f"{num:.1f} GB"


def _download_stage(cur: float, tot: float) -> str:
    """Stage label with live byte progress so a slow download reads as moving,
    not frozen, even while the rounded percentage holds steady."""
    if tot <= 0:
        return "Downloading model"
    return f"Downloading model ({_human_bytes(cur)} / {_human_bytes(tot)})"


def ensure_repo_downloaded(
    repo_id: str,
    on_progress: Callable[[str, float, float | None], None] | None = None,
) -> None:
    """Download a HuggingFace repo snapshot when missing, reporting download progress."""
    from huggingface_hub import snapshot_download

    if not on_progress:
        if _repo_in_cache(repo_id):
            return
        snapshot_download(repo_id)
        return

    # huggingface_hub creates the real per-file byte bars from
    # `huggingface_hub.utils.tqdm.tqdm` — NOT from snapshot_download's
    # `tqdm_class` argument (which only wraps the outer "Fetching N files"
    # counter). Patch that class so we observe actual byte progress.
    # NB: `import huggingface_hub.utils.tqdm as m` binds the `tqdm` *class*, not
    # the submodule, because `huggingface_hub.utils.__init__` does
    # `from .tqdm import tqdm`, rebinding that attribute on the package. Import
    # the module explicitly so we can patch its module-level `tqdm`.
    hf_tqdm_mod = importlib.import_module("huggingface_hub.utils.tqdm")

    base_tqdm = hf_tqdm_mod.tqdm

    state = {"frac": 0.0, "eta": None, "cur": 0.0, "tot": 0.0}
    state_lock = threading.Lock()
    done = threading.Event()
    errors: list[BaseException] = []
    # Aggregate progress across the per-file byte bars.
    bars: dict[int, tuple[float, float]] = {}
    # Sliding window of (monotonic_time, bytes_downloaded) for a *recent*
    # throughput estimate. A whole-run average over-/under-shoots badly on a
    # slow tail, making the ETA jump around and the bar look frozen.
    samples: deque[tuple[float, float]] = deque(maxlen=64)
    on_progress("Downloading model", 0.0, None)

    def _recompute() -> None:
        cur = sum(c for c, _ in bars.values())
        tot = sum(t for _, t in bars.values())
        frac = min(cur / tot, 1.0) if tot else 0.0
        now = time.monotonic()
        samples.append((now, cur))
        # Speed over the sampling window (≥1s span) → ETA on remaining bytes.
        eta: float | None = None
        if len(samples) >= 2:
            t0, b0 = samples[0]
            dt, db = now - t0, cur - b0
            if dt >= 1.0 and db > 0:
                eta = (tot - cur) / (db / dt)
        state.update(frac=frac, eta=eta, cur=cur, tot=tot)

    class DownloadProgress(base_tqdm):
        # Suppress terminal output without disabling tracking: `disable=True`
        # makes tqdm.update() early-return, freezing self.n at 0.
        def refresh(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return None

        def display(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return None

        def update(self, n=1):  # noqa: ANN001
            displayed = super().update(n)
            # Only byte-unit bars are real file downloads; ignore the "Fetching
            # N files" counter so the fraction reflects downloaded bytes.
            if getattr(self, "unit", "") == "B" and self.total:
                with state_lock:
                    bars[id(self)] = (float(self.n or 0.0), float(self.total))
                    _recompute()
            return displayed

    def download() -> None:
        hf_tqdm_mod.tqdm = DownloadProgress
        try:
            snapshot_download(repo_id, tqdm_class=DownloadProgress)
        except BaseException as exc:
            errors.append(exc)
        finally:
            hf_tqdm_mod.tqdm = base_tqdm
            done.set()

    thread = threading.Thread(target=download, name="hf-model-download", daemon=True)
    thread.start()

    while not done.wait(0.2):
        with state_lock:
            frac, eta, cur, tot = (
                float(state["frac"]),
                state["eta"],
                float(state["cur"]),
                float(state["tot"]),
            )
        on_progress(_download_stage(cur, tot), frac, eta)

    thread.join()
    if errors:
        raise errors[0]

    on_progress("Downloading model", 1.0, 0.0)


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

    engine: str = "whisper"
    whisper_model: str = "large-v3"
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    language: str | None = "ru"

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
