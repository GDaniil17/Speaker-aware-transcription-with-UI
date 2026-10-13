"""FunAudioLLM SenseVoiceSmall via FunASR (zh, yue, en, ja, ko)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from speaker_transcribe.audio import wav_duration
from speaker_transcribe.catalog import SENSEVOICE_LANGUAGES, SENSEVOICE_REPO, SENSEVOICE_VAD_REPO
from speaker_transcribe.config import Settings
from speaker_transcribe.engines.common import (
    ProgressBarFn,
    ProgressFn,
    cached_model,
    log,
    module_available,
)
from speaker_transcribe.hub import ensure_repo_downloaded
from speaker_transcribe.transcript import TranscriptSegment, emit_segment, group_words


def transcribe(
    wav_path: Path,
    settings: Settings,
    diar: list[dict],
    sp_map: dict,
    progress: ProgressFn | None,
    on_progress: ProgressBarFn | None = None,
) -> Iterator[TranscriptSegment]:
    """Transcribe with FunAudioLLM SenseVoiceSmall via FunASR.

    SenseVoice covers 5 languages (zh/yue/en/ja/ko) and returns word-level
    CTC-alignment timestamps, which are grouped into readable segments before
    speaker attribution.
    """
    if not module_available("funasr"):
        raise RuntimeError(
            "SenseVoice needs the funasr package. "
            "Install it with: uv sync --inexact --extra sensevoice"
        )
    from funasr import AutoModel

    log(progress, "Loading SenseVoice model…")
    ensure_repo_downloaded(SENSEVOICE_REPO, on_progress)
    ensure_repo_downloaded(SENSEVOICE_VAD_REPO, on_progress)
    if on_progress:
        on_progress("Transcribing (SenseVoice, no live progress)", 0.0, None)
    model = cached_model(
        ("sensevoice", SENSEVOICE_REPO),
        lambda: AutoModel(
            model=SENSEVOICE_REPO,
            vad_model=SENSEVOICE_VAD_REPO,
            vad_kwargs={"max_single_segment_time": 30000},
            hub="hf",
            disable_update=True,
        ),
    )

    language = settings.language if settings.language in SENSEVOICE_LANGUAGES else "auto"
    log(progress, "Transcribing with SenseVoice…")
    res = model.generate(
        input=str(wav_path),
        cache={},
        language=language,
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15,
        output_timestamp=True,
    )

    first = res[0] if res else {}
    words = _merge_subwords(_sensevoice_words(first))
    if words:
        for start, end, text in group_words(words):
            seg = emit_segment(start, end, text, diar, sp_map)
            if seg:
                yield seg
    elif first.get("text"):
        # No timestamps came back — emit the whole utterance as one segment.
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        text = rich_transcription_postprocess(first["text"])
        seg = emit_segment(0.0, wav_duration(wav_path), text, diar, sp_map)
        if seg:
            yield seg

    if on_progress:
        on_progress("Transcribing", 1.0, 0.0)


def _sensevoice_words(result: dict) -> list[tuple[float, float, str]]:
    """Pair SenseVoice `words` with `timestamp` ([start_ms, end_ms]) as second tuples."""
    words = result.get("words") or []
    stamps = result.get("timestamp") or []
    out: list[tuple[float, float, str]] = []
    for token, ts in zip(words, stamps, strict=False):
        if not ts or len(ts) < 2:
            continue
        out.append((ts[0] / 1000.0, ts[1] / 1000.0, str(token)))
    return out


def _is_cjk(text: str) -> bool:
    """True if the token holds CJK script, which SenseVoice emits one char per
    token *without* a `▁` boundary marker (unlike Latin subword pieces)."""
    return any(
        "　" <= c <= "〿"  # CJK symbols & punctuation (。、…)
        or "぀" <= c <= "ヿ"  # hiragana + katakana
        or "㐀" <= c <= "鿿"  # CJK unified ideographs
        or "가" <= c <= "힣"  # hangul syllables
        or "＀" <= c <= "￯"  # fullwidth forms (！？etc.)
        for c in text
    )


def _merge_subwords(tokens: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Join SentencePiece subwords into words (a `▁` marks a new word boundary).

    CJK languages have no `▁` markers — each character is its own token — so
    those stay separate; only Latin subword pieces (no `▁`, no CJK) get joined
    onto the previous word.
    """
    merged: list[list] = []
    for start, end, text in tokens:
        continuation = (
            merged and not text.startswith("▁") and not _is_cjk(text) and not _is_cjk(merged[-1][2])
        )
        if continuation:
            merged[-1][1] = end
            merged[-1][2] += text
        else:
            merged.append([start, end, text.lstrip("▁")])
    return [(start, end, text) for start, end, text in merged if text.strip()]
