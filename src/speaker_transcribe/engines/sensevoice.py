"""FunAudioLLM SenseVoiceSmall via FunASR (zh, yue, en, ja, ko)."""

from __future__ import annotations

import time
import unicodedata
from collections.abc import Iterator
from pathlib import Path

from speaker_transcribe.audio import read_wav_samples
from speaker_transcribe.catalog import SENSEVOICE_LANGUAGES, SENSEVOICE_REPO, VAD_FILES, VAD_REPO
from speaker_transcribe.config import Settings
from speaker_transcribe.engines.common import (
    ProgressBarFn,
    ProgressFn,
    cached_model,
    log,
    module_available,
)
from speaker_transcribe.engines.onnx import SAMPLE_RATE, speech_pieces
from speaker_transcribe.hub import ensure_repo_downloaded
from speaker_transcribe.transcript import TranscriptSegment, split_by_speaker


def _is_punctuation(text: str) -> bool:
    return all(unicodedata.category(c).startswith("P") for c in text)


def timed_words(
    words: list[str], stamps: list[list[int]], offset: float
) -> list[tuple[float, float, str]]:
    """SenseVoice `words` and `timestamp` ([start_ms, end_ms]) as (start, end, word).

    FunASR returns whole words, but punctuation and the parts of a contraction
    come as separate entries ("we", "'", "re", ","); those are joined back onto
    the word before them. CJK text comes one character per entry.
    """
    out: list[tuple[float, float, str]] = []
    glue_next = False
    for word, stamp in zip(words, stamps, strict=False):
        word = str(word).replace("▁", " ").strip()
        if not word or not stamp or len(stamp) < 2:
            continue
        start, end = offset + stamp[0] / 1000.0, offset + stamp[1] / 1000.0
        if out and (glue_next or _is_punctuation(word)):
            prev_start, _, prev_word = out[-1]
            out[-1] = (prev_start, end, prev_word + word)
        else:
            out.append((start, end, word))
        glue_next = word in ("'", "’")
    return out


def transcribe(
    wav_path: Path,
    settings: Settings,
    diar: list[dict],
    sp_map: dict,
    progress: ProgressFn | None,
    on_progress: ProgressBarFn | None = None,
) -> Iterator[TranscriptSegment]:
    """Transcribe with SenseVoiceSmall, one VAD piece (<=20 s) at a time.

    Fed long stretches at once (FunASR's own VAD merges up to ~30 s), the model
    drops whole phrases; short pieces come back complete, with punctuation.
    """
    if not module_available("funasr"):
        raise RuntimeError(
            "SenseVoice needs the funasr package. "
            "Install it with: uv sync --inexact --extra sensevoice"
        )
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    log(progress, "Loading SenseVoice model…")
    ensure_repo_downloaded(SENSEVOICE_REPO, on_progress)
    ensure_repo_downloaded(VAD_REPO, on_progress, VAD_FILES)
    model = cached_model(
        ("sensevoice", SENSEVOICE_REPO),
        lambda: AutoModel(model=SENSEVOICE_REPO, hub="hf", disable_update=True),
    )
    language = settings.language if settings.language in SENSEVOICE_LANGUAGES else "auto"

    samples = read_wav_samples(wav_path)
    if on_progress:
        on_progress("Transcribing (finding speech)", 0.0, None)
    spans = speech_pieces(samples)
    log(progress, f"Transcribing {len(spans)} speech piece(s) with SenseVoice…")

    started = time.monotonic()
    for done, (lo, hi) in enumerate(spans, 1):
        result = model.generate(
            input=samples[lo:hi],
            cache={},
            language=language,
            use_itn=True,
            output_timestamp=True,
        )
        first = result[0] if result else {}
        text = rich_transcription_postprocess(first.get("text") or "").strip()
        if text:
            start, end = lo / SAMPLE_RATE, hi / SAMPLE_RATE
            words = timed_words(first.get("words") or [], first.get("timestamp") or [], start)
            yield from split_by_speaker(start, end, text, words, diar, sp_map)
        if on_progress:
            frac = done / len(spans)
            elapsed = time.monotonic() - started
            on_progress("Transcribing", frac, elapsed / frac - elapsed)

    if on_progress:
        on_progress("Transcribing", 1.0, 0.0)
