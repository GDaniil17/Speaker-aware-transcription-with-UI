"""Sber GigaAM v3 and v2 (Russian) on ONNX Runtime via onnx-asr."""

from __future__ import annotations

import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from speaker_transcribe.audio import read_wav_samples
from speaker_transcribe.catalog import (
    GIGAAM_FILES,
    GIGAAM_MODELS,
    GIGAAM_QUANTIZATION,
    GIGAAM_VAD_FILES,
    GIGAAM_VAD_REPO,
    gigaam_repo_for,
)
from speaker_transcribe.config import Settings
from speaker_transcribe.engines.common import (
    ProgressBarFn,
    ProgressFn,
    cached_model,
    log,
    module_available,
)
from speaker_transcribe.hub import ensure_repo_downloaded
from speaker_transcribe.runtime import perf_cores
from speaker_transcribe.transcript import TranscriptSegment, split_by_speaker


def token_words(
    tokens: list[str], stamps: list[float], offset: float, end: float
) -> list[tuple[float, float, str]]:
    """Group timed tokens into (start, end, word) tuples.

    onnx-asr turns the `▁` word marker into a space. GigaAM v2/v3 emit one token
    per character with a bare space between words; the v3 e2e models emit
    subwords where a leading space starts a word (" на") and punctuation sticks
    to the word before it ("."). A word ends where the next one starts (or at
    `end`, the end of the piece).
    """
    words: list[tuple[float, str]] = []
    text = ""
    word_start = 0.0
    for token, stamp in zip(tokens, stamps, strict=False):
        if token[:1].isspace():
            if text:
                words.append((word_start, text))
                text = ""
            token = token.strip()
            if not token:
                continue
        if not text:
            word_start = offset + stamp
        text += token
    if text:
        words.append((word_start, text))
    ends = [start for start, _ in words[1:]] + [end]
    return [(start, stop, word) for (start, word), stop in zip(words, ends, strict=False)]


# Silero VAD splits at any 100 ms pause, so a sentence with a few commas falls
# into pieces that are each decoded alone (and, with the e2e models, each end
# with a full stop). Pieces closer than this are decoded together, up to the
# 20 s a GigaAM model accepts; a speaker change inside one is still split per word.
_MERGE_GAP_SECONDS = 0.5
_MAX_PIECE_SECONDS = 20.0


def merge_spans(
    spans: list[tuple[int, int]],
    rate: int = 16000,
    max_gap: float = _MERGE_GAP_SECONDS,
    max_len: float = _MAX_PIECE_SECONDS,
) -> list[tuple[int, int]]:
    """Join speech spans (in samples) separated by short pauses."""
    merged: list[tuple[int, int]] = []
    for lo, hi in spans:
        if merged:
            prev_lo, prev_hi = merged[-1]
            if lo - prev_hi <= max_gap * rate and hi - prev_lo <= max_len * rate:
                merged[-1] = (prev_lo, hi)
                continue
        merged.append((lo, hi))
    return merged


def transcribe(
    wav_path: Path,
    settings: Settings,
    diar: list[dict],
    sp_map: dict,
    progress: ProgressFn | None,
    on_progress: ProgressBarFn | None = None,
) -> Iterator[TranscriptSegment]:
    """Transcribe Russian audio with Sber GigaAM (v3 or v2) via onnx-asr.

    Silero VAD cuts the audio into pieces of at most 20 s (the model's limit).
    The pieces are decoded in parallel, one single-threaded ONNX session per
    performance core: about 3x faster than one session spread over all cores,
    with identical text. The v3 e2e models write punctuation and capitals; the
    others lowercase words only.
    """
    if not module_available("onnx_asr"):
        raise RuntimeError("GigaAM needs the onnx-asr package. Install it with: uv sync")
    import numpy as np
    import onnx_asr
    import onnxruntime as rt

    model_name = settings.whisper_model
    if model_name not in GIGAAM_MODELS:
        model_name = next(iter(GIGAAM_MODELS))
    log(progress, f"Loading GigaAM model ({model_name})…")
    ensure_repo_downloaded(gigaam_repo_for(model_name), on_progress, GIGAAM_FILES[model_name])
    ensure_repo_downloaded(GIGAAM_VAD_REPO, on_progress, GIGAAM_VAD_FILES)

    def build():
        # CPU on purpose: the CoreML provider onnx-asr picks on a Mac is slower
        # for these models and crashed on long audio in testing.
        cpu = ["CPUExecutionProvider"]
        opts = rt.SessionOptions()
        opts.intra_op_num_threads = 1
        vad = onnx_asr.load_vad("silero", providers=cpu)
        asr = onnx_asr.load_model(
            model_name, quantization=GIGAAM_QUANTIZATION, providers=cpu, sess_options=opts
        )
        return vad, asr.with_timestamps()

    vad, asr = cached_model(("gigaam", model_name), build)

    samples = read_wav_samples(wav_path)
    if on_progress:
        on_progress("Transcribing (finding speech)", 0.0, None)
    spans = merge_spans(
        list(next(vad.segment_batch(samples[None, :], np.array([len(samples)]), 16000)))
    )
    log(progress, f"Transcribing {len(spans)} speech piece(s) with GigaAM…")

    def decode(span: tuple[int, int]):
        return asr.recognize(samples[span[0] : span[1]], sample_rate=16000)

    results: list = [None] * len(spans)
    started = time.monotonic()
    with ThreadPoolExecutor(perf_cores()) as pool:
        futures = {pool.submit(decode, span): i for i, span in enumerate(spans)}
        for done, fut in enumerate(as_completed(futures), 1):
            results[futures[fut]] = fut.result()
            if on_progress:
                frac = done / len(spans)
                elapsed = time.monotonic() - started
                on_progress("Transcribing", frac, elapsed / frac - elapsed)

    for (lo, hi), res in zip(spans, results, strict=True):
        start, end = lo / 16000, hi / 16000
        text = (res.text or "").strip()
        if not text:
            continue
        words = token_words(list(res.tokens or []), list(res.timestamps or []), start, end)
        yield from split_by_speaker(start, end, text, words, diar, sp_map)

    if on_progress:
        on_progress("Transcribing", 1.0, 0.0)
