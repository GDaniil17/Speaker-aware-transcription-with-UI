"""Core pipeline: speaker diarization (pyannote) + transcription (Whisper)."""

from __future__ import annotations

import importlib.util
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from speaker_transcribe.audio import prepare_audio
from speaker_transcribe.config import (
    PARAKEET_REPOS,
    QWEN3_ALIGNER_REPO,
    QWEN3_ASR_LANGUAGES,
    QWEN3_ASR_REPO,
    SENSEVOICE_LANGUAGES,
    SENSEVOICE_REPO,
    Settings,
    ensure_repo_downloaded,
    ensure_whisper_model_downloaded,
    is_apple_silicon,
    mlx_repo_for,
)

ProgressFn = Callable[[str], None]
ProgressBarFn = Callable[[str, float, float | None], None]


@dataclass
class TranscriptSegment:
    """One transcribed chunk attributed to a speaker."""

    start: float
    end: float
    speaker: str
    text: str


def _log(progress: ProgressFn | None, msg: str) -> None:
    if progress:
        progress(msg)


def importlib_found(name: str) -> bool:
    """True when a module can be imported (used to pick optional engine backends)."""
    return importlib.util.find_spec(name) is not None


def _make_diarization_hook(progress: ProgressFn | None, on_progress: ProgressBarFn | None):
    """Adapt pyannote's hook protocol to our progress callbacks."""
    if not progress and not on_progress:
        return None

    last = {"step": None}

    def hook(step_name, step_artifact=None, file=None, completed=None, total=None):
        if completed is not None and total:
            frac = completed / max(total, 1)
            if on_progress:
                on_progress(f"Diarizing · {step_name}", frac, None)
            if progress and step_name != last["step"]:
                progress(f"Diarizing: {step_name}…")
                last["step"] = step_name

    return hook


def _diarize(
    wav_path: Path,
    settings: Settings,
    progress: ProgressFn | None,
    on_progress: ProgressBarFn | None = None,
):
    """Run pyannote diarization, returning (segments, speaker_map)."""
    import torch
    from pyannote.audio import Pipeline as DiarizationPipeline

    if not settings.hf_token:
        raise RuntimeError(
            "A HuggingFace token is required for diarization. Set HF_TOKEN "
            "in your environment/.env and accept the model terms at "
            f"https://huggingface.co/{settings.diarization_model}"
        )

    _log(progress, "Loading diarization model…")
    pipeline = DiarizationPipeline.from_pretrained(
        settings.diarization_model, token=settings.hf_token
    )
    hook = _make_diarization_hook(progress, on_progress)

    def run_on(device: str):
        pipeline.to(torch.device(device))
        return pipeline(
            str(wav_path),
            min_speakers=settings.min_speakers,
            max_speakers=settings.max_speakers,
            hook=hook,
        )

    device = settings.diarization_device or "cpu"
    _log(progress, f"Diarizing on {device} (detecting speakers)…")
    try:
        output = run_on(device)
    except Exception as exc:
        if device == "cpu":
            raise
        _log(progress, f"{device} diarization failed ({exc}); retrying on CPU…")
        output = run_on("cpu")

    segments: list[dict] = []
    if hasattr(output, "speaker_diarization"):
        for turn, speaker in output.speaker_diarization:
            segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    elif hasattr(output, "itertracks"):
        for turn, _, speaker in output.itertracks(yield_label=True):
            segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    else:
        raise RuntimeError("Unknown pyannote output format.")

    unique = sorted({s["speaker"] for s in segments})
    speaker_map = {spk: f"Speaker {i + 1}" for i, spk in enumerate(unique)}
    _log(progress, f"Diarization done — {len(unique)} speaker(s) found.")
    return segments, speaker_map


def _speaker_for(seg_start: float, seg_end: float, diar: list[dict], sp_map: dict) -> str:
    """Attribute a transcription segment to the speaker with the most overlap."""
    best_speaker = "Unknown"
    best_overlap = 0.0
    for ds in diar:
        overlap = max(0.0, min(seg_end, ds["end"]) - max(seg_start, ds["start"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = sp_map[ds["speaker"]]
    return best_speaker


def _transcribe(
    wav_path: Path,
    settings: Settings,
    diar: list[dict],
    sp_map: dict,
    progress: ProgressFn | None,
    on_progress: ProgressBarFn | None = None,
) -> Iterator[TranscriptSegment]:
    """Dispatch to the configured transcription engine."""
    if settings.engine == "parakeet":
        yield from _transcribe_parakeet(wav_path, settings, diar, sp_map, progress, on_progress)
    elif settings.engine == "qwen3-asr":
        yield from _transcribe_qwen3(wav_path, settings, diar, sp_map, progress, on_progress)
    elif settings.engine == "sensevoice":
        yield from _transcribe_sensevoice(wav_path, settings, diar, sp_map, progress, on_progress)
    elif settings.backend == "mlx":
        yield from _transcribe_mlx(wav_path, settings, diar, sp_map, progress, on_progress)
    else:
        yield from _transcribe_faster_whisper(
            wav_path, settings, diar, sp_map, progress, on_progress
        )


def _transcribe_mlx(
    wav_path: Path,
    settings: Settings,
    diar: list[dict],
    sp_map: dict,
    progress: ProgressFn | None,
    on_progress: ProgressBarFn | None = None,
) -> Iterator[TranscriptSegment]:
    """Transcribe with Apple's MLX (single blocking call, no streaming progress)."""
    import mlx_whisper

    repo = mlx_repo_for(settings.whisper_model)
    _log(progress, f"Loading Whisper model ({settings.whisper_model})…")
    ensure_whisper_model_downloaded(settings.whisper_model, settings.backend, on_progress)
    _log(progress, f"Transcribing with MLX ({repo})…")
    if on_progress:
        on_progress("Transcribing (MLX, no live progress)", 0.0, None)
    result = mlx_whisper.transcribe(
        str(wav_path),
        path_or_hf_repo=repo,
        language=settings.language,
        temperature=settings.temperature,
        compression_ratio_threshold=settings.compression_ratio_threshold,
        logprob_threshold=settings.log_prob_threshold,
        no_speech_threshold=settings.no_speech_threshold,
        condition_on_previous_text=settings.condition_on_previous_text,
        initial_prompt=settings.initial_prompt,
        hallucination_silence_threshold=settings.hallucination_silence_threshold,
        word_timestamps=False,
        verbose=None,
    )

    segments = result.get("segments") or []
    total = segments[-1]["end"] if segments else (settings.duration or None)
    for seg in segments:
        if on_progress and total:
            on_progress("Transcribing", min(seg["end"] / total, 1.0), 0.0)
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        speaker = _speaker_for(seg["start"], seg["end"], diar, sp_map)
        yield TranscriptSegment(seg["start"], seg["end"], speaker, text)

    if on_progress:
        on_progress("Transcribing", 1.0, 0.0)


def _transcribe_faster_whisper(
    wav_path: Path,
    settings: Settings,
    diar: list[dict],
    sp_map: dict,
    progress: ProgressFn | None,
    on_progress: ProgressBarFn | None = None,
) -> Iterator[TranscriptSegment]:
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    _log(progress, f"Loading Whisper model ({settings.whisper_model})…")
    ensure_whisper_model_downloaded(settings.whisper_model, settings.backend, on_progress)
    model = WhisperModel(
        settings.whisper_model,
        device=settings.device,
        compute_type=settings.compute_type,
    )
    pipeline = BatchedInferencePipeline(model)
    pipeline.chunk_length = settings.chunk_length

    _log(progress, "Transcribing…")
    if on_progress:
        on_progress("Transcribing", 0.0, None)
    segments, info = pipeline.transcribe(
        str(wav_path),
        language=settings.language,
        beam_size=settings.beam_size,
        temperature=settings.temperature,
        compression_ratio_threshold=settings.compression_ratio_threshold,
        log_prob_threshold=settings.log_prob_threshold,
        no_speech_threshold=settings.no_speech_threshold,
        condition_on_previous_text=settings.condition_on_previous_text,
        initial_prompt=settings.initial_prompt,
        repetition_penalty=settings.repetition_penalty,
        no_repeat_ngram_size=settings.no_repeat_ngram_size,
        word_timestamps=False,
        vad_filter=settings.vad_filter,
        hallucination_silence_threshold=settings.hallucination_silence_threshold,
        batch_size=settings.batch_size,
        chunk_length=settings.chunk_length,
    )

    total = getattr(info, "duration", None) or settings.duration
    started = time.monotonic()

    for seg in segments:
        if on_progress and total:
            frac = min(seg.end / total, 1.0)
            elapsed = time.monotonic() - started
            eta = (elapsed / frac - elapsed) if frac > 0 else None
            on_progress("Transcribing", frac, eta)

        text = seg.text.strip()
        if not text:
            continue
        speaker = _speaker_for(seg.start, seg.end, diar, sp_map)
        yield TranscriptSegment(seg.start, seg.end, speaker, text)

    if on_progress:
        on_progress("Transcribing", 1.0, 0.0)


def _emit(
    start: float,
    end: float,
    text: str,
    diar: list[dict],
    sp_map: dict,
) -> TranscriptSegment | None:
    """Build a speaker-attributed segment, or None when the text is empty."""
    text = (text or "").strip()
    if not text:
        return None
    return TranscriptSegment(start, end, _speaker_for(start, end, diar, sp_map), text)


def _transcribe_parakeet(
    wav_path: Path,
    settings: Settings,
    diar: list[dict],
    sp_map: dict,
    progress: ProgressFn | None,
    on_progress: ProgressBarFn | None = None,
) -> Iterator[TranscriptSegment]:
    """Transcribe with NVIDIA Parakeet (MLX on Apple Silicon, else NeMo).

    Parakeet auto-detects its language (25 European languages); settings.language
    is ignored. Both backends yield sentence-level timestamps.
    """
    use_mlx = is_apple_silicon() and importlib_found("parakeet_mlx")
    _log(progress, "Loading Parakeet model…")
    if on_progress:
        on_progress("Transcribing (Parakeet, no live progress)", 0.0, None)

    if use_mlx:
        from parakeet_mlx import from_pretrained

        repo = PARAKEET_REPOS["mlx"]
        ensure_repo_downloaded(repo, on_progress)
        _log(progress, f"Transcribing with Parakeet MLX ({repo})…")
        model = from_pretrained(repo)
        result = model.transcribe(
            str(wav_path),
            chunk_duration=120.0,
            overlap_duration=15.0,
        )
        rows = [(s.start, s.end, s.text) for s in result.sentences]
    else:
        if not importlib_found("nemo"):
            raise RuntimeError(
                "Parakeet needs NeMo (or parakeet-mlx on Apple Silicon). "
                "Install it with: uv pip install '.[parakeet]'"
            )
        import nemo.collections.asr as nemo_asr  # type: ignore

        repo = PARAKEET_REPOS["nemo"]
        ensure_repo_downloaded(repo, on_progress)
        _log(progress, f"Transcribing with Parakeet NeMo ({repo})…")
        model = nemo_asr.models.ASRModel.from_pretrained(repo)
        out = model.transcribe([str(wav_path)], timestamps=True)
        first = out[0]
        stamp = first.timestamp if hasattr(first, "timestamp") else first["timestamp"]
        segs = stamp["segment"]
        rows = [(s["start"], s["end"], s.get("segment", s.get("text", ""))) for s in segs]

    total = rows[-1][1] if rows else (settings.duration or None)
    for start, end, text in rows:
        if on_progress and total:
            on_progress("Transcribing", min(end / total, 1.0), 0.0)
        seg = _emit(start, end, text, diar, sp_map)
        if seg:
            yield seg

    if on_progress:
        on_progress("Transcribing", 1.0, 0.0)


def _transcribe_qwen3(
    wav_path: Path,
    settings: Settings,
    diar: list[dict],
    sp_map: dict,
    progress: ProgressFn | None,
    on_progress: ProgressBarFn | None = None,
) -> Iterator[TranscriptSegment]:
    """Transcribe with Qwen3-ASR-1.7B + its forced aligner for word timestamps.

    The aligner handles up to ~5 minutes per call, so long audio is split into
    fixed windows; word timestamps are offset back to absolute time and grouped
    into sentence-like segments before speaker attribution.
    """
    if not importlib_found("qwen_asr"):
        raise RuntimeError(
            "Qwen3-ASR needs the qwen-asr package. "
            "Install it with: uv pip install '.[qwen3-asr]'"
        )
    from qwen_asr import Qwen3ASRModel

    _log(progress, "Loading Qwen3-ASR model…")
    ensure_repo_downloaded(QWEN3_ASR_REPO, on_progress)
    ensure_repo_downloaded(QWEN3_ALIGNER_REPO, on_progress)
    if on_progress:
        on_progress("Transcribing (Qwen3-ASR, no live progress)", 0.0, None)
    model = Qwen3ASRModel.from_pretrained(
        QWEN3_ASR_REPO, forced_aligner=QWEN3_ALIGNER_REPO
    )

    # Qwen wants the language *name* (e.g. "English"); None = auto-detect.
    language = QWEN3_ASR_LANGUAGES.get(settings.language) if settings.language else None

    words: list[tuple[float, float, str]] = []
    with tempfile.TemporaryDirectory(prefix="qwen3_") as tmp:
        chunks = list(_wav_windows(wav_path, Path(tmp), window_seconds=240))
        for i, (chunk_path, chunk_start) in enumerate(chunks):
            if on_progress and chunks:
                on_progress("Transcribing", min(i / len(chunks), 1.0), None)
            res = model.transcribe(
                audio=str(chunk_path), language=language, return_time_stamps=True
            )
            if not res:
                continue
            for ts in res[0].time_stamps or []:
                words.append(
                    (
                        chunk_start + float(ts.start_time),
                        chunk_start + float(ts.end_time),
                        str(ts.text),
                    )
                )

    for start, end, text in _group_words(words):
        seg = _emit(start, end, text, diar, sp_map)
        if seg:
            yield seg

    if on_progress:
        on_progress("Transcribing", 1.0, 0.0)


def _wav_windows(
    wav_path: Path,
    out_dir: Path,
    window_seconds: int = 240,
) -> Iterator[tuple[Path, float]]:
    """Split a mono 16 kHz WAV into fixed windows, yielding (path, start_seconds)."""
    import wave

    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        total = wf.getnframes()
        window = window_seconds * sr
        for index, offset in enumerate(range(0, max(total, 1), window)):
            wf.setpos(offset)
            frames = wf.readframes(min(window, total - offset))
            if not frames:
                break
            out = out_dir / f"window_{index:04d}.wav"
            with wave.open(str(out), "wb") as ww:
                ww.setnchannels(channels)
                ww.setsampwidth(sampwidth)
                ww.setframerate(sr)
                ww.writeframes(frames)
            yield out, offset / sr


def _transcribe_sensevoice(
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
    if not importlib_found("funasr"):
        raise RuntimeError(
            "SenseVoice needs the funasr package. "
            "Install it with: uv pip install '.[sensevoice]'"
        )
    from funasr import AutoModel

    _log(progress, "Loading SenseVoice model…")
    ensure_repo_downloaded(SENSEVOICE_REPO, on_progress)
    if on_progress:
        on_progress("Transcribing (SenseVoice, no live progress)", 0.0, None)
    model = AutoModel(
        model=SENSEVOICE_REPO,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        hub="hf",
        disable_update=True,
    )

    language = settings.language if settings.language in SENSEVOICE_LANGUAGES else "auto"
    _log(progress, "Transcribing with SenseVoice…")
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
        for start, end, text in _group_words(words):
            seg = _emit(start, end, text, diar, sp_map)
            if seg:
                yield seg
    elif first.get("text"):
        # No timestamps came back — emit the whole utterance as one segment.
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        text = rich_transcription_postprocess(first["text"])
        seg = _emit(0.0, _wav_duration(wav_path), text, diar, sp_map)
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
            merged
            and not text.startswith("▁")
            and not _is_cjk(text)
            and not _is_cjk(merged[-1][2])
        )
        if continuation:
            merged[-1][1] = end
            merged[-1][2] += text
        else:
            merged.append([start, end, text.lstrip("▁")])
    return [(start, end, text) for start, end, text in merged if text.strip()]


def _wav_duration(wav_path: Path) -> float:
    """Duration of a WAV file in seconds."""
    import wave

    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate() or 16000)


def _join_tokens(tokens: list[str]) -> str:
    """Join ASR tokens into text — spaces for word tokens, none for CJK characters."""
    word_like = any(len(t.strip()) > 1 and t.strip().isalpha() for t in tokens)
    sep = " " if word_like else ""
    return sep.join(t.strip() for t in tokens if t.strip()).strip()


def _group_words(
    words: list[tuple[float, float, str]],
    gap: float = 0.6,
    max_chars: int = 240,
) -> Iterator[tuple[float, float, str]]:
    """Merge word/char timestamps into readable segments.

    Breaks on sentence-ending punctuation, a silence gap longer than `gap`
    seconds, or when a segment grows past `max_chars` characters.
    """
    cur: list[tuple[float, float, str]] = []

    def flush() -> Iterator[tuple[float, float, str]]:
        if cur:
            yield cur[0][0], cur[-1][1], _join_tokens([w[2] for w in cur])

    for i, w in enumerate(words):
        cur.append(w)
        text_so_far = "".join(x[2] for x in cur)
        ends_sentence = w[2].strip().endswith((".", "?", "!", "。", "？", "！", "…"))
        next_gap = words[i + 1][0] - w[1] if i + 1 < len(words) else 0.0
        if ends_sentence or next_gap > gap or len(text_so_far) >= max_chars:
            yield from flush()
            cur = []
    yield from flush()


def run_pipeline(
    settings: Settings,
    progress: ProgressFn | None = None,
    on_progress: ProgressBarFn | None = None,
) -> list[TranscriptSegment]:
    """End-to-end: prepare audio, diarize, transcribe, attribute speakers."""
    with tempfile.TemporaryDirectory(prefix="spk_") as tmp:
        wav_path = Path(tmp) / "input.wav"
        _log(progress, "Preparing audio…")
        prepare_audio(
            settings.input_path,
            wav_path,
            start=settings.start,
            duration=settings.duration,
        )

        if settings.diarize:
            diar, sp_map = _diarize(wav_path, settings, progress, on_progress)
        else:
            _log(progress, "Skipping diarization — transcription only.")
            diar, sp_map = [], {}
        result = list(_transcribe(wav_path, settings, diar, sp_map, progress, on_progress))

        if not settings.diarize:
            for seg in result:
                seg.speaker = "Speaker 1"

    if settings.start:
        for seg in result:
            seg.start += settings.start
            seg.end += settings.start

    _log(progress, f"Done — {len(result)} transcript segment(s).")
    return result
