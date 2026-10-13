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


def wav_duration(wav_path: str | Path) -> float:
    """Duration of a WAV file in seconds."""
    import wave

    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate() or 16000)


def slice_wav(
    wav_path: str | Path,
    out_path: str | Path,
    start: float = 0.0,
    end: float | None = None,
) -> Path:
    """Copy `[start, end)` of a WAV into a new file — a frame copy, no re-encoding.

    Used to cut an already-prepared WAV into chunks, so it stays cheap even when
    the stall fallback has to re-cut the same audio at several chunk sizes.
    """
    import wave

    with wave.open(str(wav_path), "rb") as wf:
        params = wf.getparams()
        rate = params.framerate or 16000
        total = params.nframes
        first = min(max(int(start * rate), 0), total)
        last = total if end is None else min(max(int(end * rate), first), total)
        wf.setpos(first)
        frames = wf.readframes(last - first)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as out:
        out.setnchannels(params.nchannels)
        out.setsampwidth(params.sampwidth)
        out.setframerate(params.framerate)
        out.writeframes(frames)
    return out_path


def read_wav_samples(wav_path: str | Path):
    """Samples of a prepared 16 kHz mono s16 WAV as a float32 array in [-1, 1].

    Engines that accept raw audio get it this way instead of re-decoding the
    file with an ffmpeg binary of their own.
    """
    import wave

    import numpy as np

    with wave.open(str(wav_path), "rb") as wf:
        if wf.getframerate() != 16000 or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise RuntimeError("Expected 16 kHz mono 16-bit audio.")
        data = wf.readframes(wf.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
