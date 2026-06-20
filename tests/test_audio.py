import subprocess
import wave
from pathlib import Path

from speaker_transcribe.audio import get_ffmpeg, prepare_audio


def _make_tone(path: Path, seconds: int) -> None:
    subprocess.run(
        [
            get_ffmpeg(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def test_prepare_audio_trims_and_converts(tmp_path):
    src = tmp_path / "tone.wav"
    _make_tone(src, seconds=5)

    out = prepare_audio(src, tmp_path / "out.wav", start=1.0, duration=2.0)
    assert out.exists()

    with wave.open(str(out)) as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 16000
        assert 1.5 < w.getnframes() / w.getframerate() < 2.5
