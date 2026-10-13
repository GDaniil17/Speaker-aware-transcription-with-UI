"""Engines that run on ONNX Runtime through onnx-asr (GigaAM, Parakeet).

Silero VAD cuts the audio into speech pieces, short pauses are merged back, and
the pieces are decoded in parallel with word timestamps, so a speaker change
inside a piece is still split per word.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cache
from pathlib import Path

from speaker_transcribe.audio import read_wav_samples
from speaker_transcribe.engines.common import ProgressBarFn, ProgressFn, cached_model, log
from speaker_transcribe.runtime import perf_cores
from speaker_transcribe.transcript import TranscriptSegment, split_by_speaker

SAMPLE_RATE = 16000

# Silero VAD splits at any 100 ms pause, so a sentence with a few commas falls
# into pieces that are each decoded alone (and, with punctuating models, each
# end with a full stop). Pieces closer than this are decoded together, up to the
# 20 s these models take in one go; a speaker change inside one is still split
# per word.
_MERGE_GAP_SECONDS = 0.5
_MAX_PIECE_SECONDS = 20.0


def token_words(
    tokens: list[str], stamps: list[float], offset: float, end: float
) -> list[tuple[float, float, str]]:
    """Group timed tokens into (start, end, word) tuples.

    onnx-asr turns the `▁` word marker into a space. GigaAM v2/v3 emit one token
    per character with a bare space between words; subword models (GigaAM v3
    e2e, Parakeet) emit pieces where a leading space starts a word (" на") and
    punctuation sticks to the word before it ("."). A word ends where the next
    one starts (or at `end`, the end of the piece).
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


def merge_spans(
    spans: list[tuple[int, int]],
    rate: int = SAMPLE_RATE,
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


# CPU on purpose: the CoreML provider onnx-asr picks on a Mac is slower for
# these models and crashed on long audio in testing.
_CPU = ["CPUExecutionProvider"]


@cache
def _silero_vad():
    import onnx_asr

    return onnx_asr.load_vad("silero", providers=_CPU)


def speech_pieces(samples) -> list[tuple[int, int]]:
    """Speech spans (in samples) of 16 kHz audio: Silero VAD, short pauses merged."""
    import numpy as np

    if len(samples) < SAMPLE_RATE // 10:
        return []  # under 0.1 s: no speech, and too short for the VAD's window

    found = next(
        _silero_vad().segment_batch(samples[None, :], np.array([len(samples)]), SAMPLE_RATE)
    )
    return merge_spans(list(found))


def transcribe_onnx(
    model_name: str,
    quantization: str | None,
    label: str,
    wav_path: Path,
    diar: list[dict],
    sp_map: dict,
    progress: ProgressFn | None,
    on_progress: ProgressBarFn | None = None,
) -> Iterator[TranscriptSegment]:
    """Transcribe with onnx-asr model `model_name`; its files must be downloaded.

    The pieces are decoded in parallel, one single-threaded ONNX session run per
    performance core: about 3x faster than one run spread over all cores, with
    identical text.
    """
    import onnx_asr
    import onnxruntime as rt

    def build():
        opts = rt.SessionOptions()
        opts.intra_op_num_threads = 1
        asr = onnx_asr.load_model(
            model_name, quantization=quantization, providers=_CPU, sess_options=opts
        )
        return asr.with_timestamps()

    asr = cached_model(("onnx-asr", model_name, quantization), build)

    samples = read_wav_samples(wav_path)
    if on_progress:
        on_progress("Transcribing (finding speech)", 0.0, None)
    spans = speech_pieces(samples)
    log(progress, f"Transcribing {len(spans)} speech piece(s) with {label}…")

    def decode(span: tuple[int, int]):
        return asr.recognize(samples[span[0] : span[1]], sample_rate=SAMPLE_RATE)

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
        start, end = lo / SAMPLE_RATE, hi / SAMPLE_RATE
        text = (res.text or "").strip()
        if not text:
            continue
        words = token_words(list(res.tokens or []), list(res.timestamps or []), start, end)
        yield from split_by_speaker(start, end, text, words, diar, sp_map)

    if on_progress:
        on_progress("Transcribing", 1.0, 0.0)
