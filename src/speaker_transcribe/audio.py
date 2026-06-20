"""Audio preparation: extract a fragment and convert to 16 kHz mono WAV."""

from __future__ import annotations

import subprocess
from pathlib import Path


def get_ffmpeg() -> str:
    """Return a usable ffmpeg binary, preferring the system one."""
    from shutil import which

    system = which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg not found. Install it (e.g. `brew install ffmpeg`) or `uv add imageio-ffmpeg`."
        ) from exc


def prepare_audio(
    input_path: str | Path,
    out_path: str | Path,
    start: float = 0.0,
    duration: float | None = None,
) -> Path:
    """Convert any audio/video file to mono 16 kHz s16 WAV, optionally a slice."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [get_ffmpeg(), "-y"]
    if start and start > 0:
        cmd += ["-ss", str(start)]
    if duration and duration > 0:
        cmd += ["-t", str(duration)]
    cmd += [
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(out_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-2000:]}")
    return out_path
