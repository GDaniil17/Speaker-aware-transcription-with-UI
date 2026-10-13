"""Where transcription runs: device and backend detection, installed engines."""

from __future__ import annotations

import importlib.util
import os
import platform
import subprocess
import sys


def is_apple_silicon() -> bool:
    """True on an M-series Mac, where MLX can use the GPU."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def mlx_available() -> bool:
    """True when the mlx-whisper backend can be used (installed, Apple Silicon)."""
    return is_apple_silicon() and importlib.util.find_spec("mlx_whisper") is not None


def faster_whisper_available() -> bool:
    """True when the faster-whisper backend is installed."""
    return importlib.util.find_spec("faster_whisper") is not None


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


def perf_cores() -> int:
    """Performance cores to run CPU models on.

    Apple Silicon efficiency cores only slow a pool of model threads down, so
    count the performance cores there; elsewhere use the physical cores.
    """
    for key in ("hw.perflevel0.physicalcpu", "hw.physicalcpu"):
        try:
            out = subprocess.run(
                ["sysctl", "-n", key], capture_output=True, text=True, timeout=5
            ).stdout.strip()
            if int(out) > 0:
                return int(out)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return max((os.cpu_count() or 2) // 2, 1)


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


def hardware_profile() -> str:
    """Detected acceleration profile for recommendations: 'cuda', 'mlx', or 'cpu'."""
    if detect_device()[0] == "cuda":
        return "cuda"
    if is_apple_silicon():
        return "mlx"
    return "cpu"


def parakeet_backend() -> str | None:
    """Backend Parakeet runs on: the fastest one installed, or None.

    'mlx' (parakeet-mlx, Apple Silicon GPU) and 'nemo' (NVIDIA NeMo, CUDA) are
    optional extras; 'onnx' (onnx-asr, CPU) is a core dependency, so Parakeet
    works without installing anything.
    """
    if is_apple_silicon() and importlib.util.find_spec("parakeet_mlx") is not None:
        return "mlx"
    if importlib.util.find_spec("nemo") is not None:
        return "nemo"
    if importlib.util.find_spec("onnx_asr") is not None:
        return "onnx"
    return None


PARAKEET_BACKEND_LABELS = {
    "mlx": "MLX (Apple Silicon GPU)",
    "nemo": "NeMo",
    "onnx": "ONNX (CPU)",
}


def parakeet_available() -> bool:
    """True when some Parakeet backend is installed (always, with the core deps)."""
    return parakeet_backend() is not None


def qwen3_asr_available() -> bool:
    """True when the qwen-asr package is importable."""
    return importlib.util.find_spec("qwen_asr") is not None


def sensevoice_available() -> bool:
    """True when the funasr package (SenseVoice backend) is importable."""
    return importlib.util.find_spec("funasr") is not None


def gigaam_available() -> bool:
    """True when onnx-asr (the GigaAM backend) is importable."""
    return importlib.util.find_spec("onnx_asr") is not None


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
    if engine == "gigaam":
        return gigaam_available()
    return False
