import subprocess
import wave
from pathlib import Path

import pytest

from speaker_transcribe.audio import get_ffmpeg, prepare_audio, slice_wav, wav_duration


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


def test_slice_wav_cuts_a_span_without_re_encoding(tmp_path):
    src = tmp_path / "tone.wav"
    _make_tone(src, seconds=10)
    prepared = prepare_audio(src, tmp_path / "prepared.wav")

    piece = slice_wav(prepared, tmp_path / "chunk.wav", start=4.0, end=7.0)

    assert wav_duration(prepared) == pytest.approx(10.0, abs=0.1)
    assert wav_duration(piece) == pytest.approx(3.0, abs=0.05)
    with wave.open(str(piece)) as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 16000


def test_slice_wav_clamps_to_the_end_of_the_file(tmp_path):
    src = tmp_path / "tone.wav"
    _make_tone(src, seconds=3)
    prepared = prepare_audio(src, tmp_path / "prepared.wav")

    piece = slice_wav(prepared, tmp_path / "chunk.wav", start=2.0, end=99.0)

    assert wav_duration(piece) == pytest.approx(1.0, abs=0.05)
