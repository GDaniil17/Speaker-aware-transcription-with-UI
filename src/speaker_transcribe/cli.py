"""Command-line entry point — useful for quick tests on short fragments.

Example:
    uv run transcribe --input sample-interview.mp3 --duration 60 --model small
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from speaker_transcribe.catalog import DEFAULT_WHISPER_MODEL, ENGINES
from speaker_transcribe.chunking import (
    DEFAULT_CHUNK_LADDER,
    DEFAULT_STALL_TIMEOUT,
    format_ladder,
    parse_ladder,
)
from speaker_transcribe.config import Settings
from speaker_transcribe.engine_install import engine_install_hint
from speaker_transcribe.export import EXPORTERS, export
from speaker_transcribe.pipeline import run_pipeline
from speaker_transcribe.runtime import describe_runtime, engine_available


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transcribe",
        description="Speaker-aware transcription (Whisper and other local ASR engines + pyannote).",
    )
    p.add_argument("--input", help="Path to an audio/video file.")
    p.add_argument("--start", type=float, default=0.0, help="Start offset (seconds).")
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Process only this many seconds (great for testing).",
    )
    p.add_argument(
        "--engine",
        default="whisper",
        choices=list(ENGINES),
        help="ASR engine. whisper (default), parakeet (NVIDIA, 25 EU langs), "
        "qwen3-asr (Qwen3-ASR-1.7B, 30 langs), sensevoice, gigaam (GigaAM v3/v2, Russian). "
        "Some engines need an extra install — see --list-models.",
    )
    p.add_argument(
        "--model",
        default=None,
        metavar="NAME",
        help="Model id for the chosen engine (default: engine's default model). "
        "For Whisper: tiny, base, small, medium, large-v3, large-v3-turbo, turbo, "
        "distil-*. See --list-models.",
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        help="List engines and their models with language coverage, then exit.",
    )
    p.add_argument(
        "--language",
        default="auto",
        help="Language code (e.g. en, ru), or 'auto' to detect it (default).",
    )
    p.add_argument(
        "--no-diarize",
        action="store_true",
        help="Transcription only, no speaker labels (no pyannote, no HF token).",
    )
    p.add_argument("--min-speakers", type=int, default=2)
    p.add_argument("--max-speakers", type=int, default=5)
    p.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "mlx", "faster-whisper"],
        help="Transcription backend. auto = MLX on Apple Silicon, else faster-whisper.",
    )
    p.add_argument(
        "--beam-size",
        default="auto",
        help="Beam size, or 'auto' (1 on CPU, 5 on GPU). Higher = slower, more accurate.",
    )
    p.add_argument(
        "--stall-timeout",
        type=float,
        default=DEFAULT_STALL_TIMEOUT,
        metavar="SECONDS",
        help="Give up on a transcription that has made no progress for this many "
        f"seconds and retry it in chunks (default: {DEFAULT_STALL_TIMEOUT:g}). "
        "0 = never interrupt a run (no stall recovery).",
    )
    p.add_argument(
        "--chunk-ladder",
        default=format_ladder(DEFAULT_CHUNK_LADDER),
        metavar="MINUTES",
        help="Chunk sizes in minutes, tried in order after a stall "
        f"(default: '{format_ladder(DEFAULT_CHUNK_LADDER)}').",
    )
    p.add_argument(
        "--no-chunk-fallback",
        action="store_true",
        help="Never chunk: a stalled run fails instead of being retried smaller.",
    )
    p.add_argument(
        "--chunk-first",
        action="store_true",
        help="Skip the whole-file attempt and start at the first chunk size "
        "(for a file you already know stalls).",
    )
    p.add_argument(
        "--fail-on-stall",
        action="store_true",
        help="Fail the run if any audio stalls at every chunk size, instead of "
        "skipping it and keeping the rest of the transcript.",
    )
    p.add_argument(
        "--formats",
        nargs="+",
        default=["txt", "srt"],
        choices=list(EXPORTERS),
        help="Output formats to write.",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Where to write outputs (default: next to the input file).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_models:
        for engine in ENGINES.values():
            extra = engine_install_hint(engine.key)
            tag = "" if engine_available(engine.key) else f"  (needs: {extra})"
            print(f"\n{engine.label}{tag} — {engine.description}")
            width = max(len(m) for m in engine.models)
            for name, desc in engine.models.items():
                print(f"  {name:<{width}}  {desc}")
        return 0

    if not args.input:
        print("Error: --input is required (or use --list-models).", file=sys.stderr)
        return 2

    engine = ENGINES[args.engine]
    if not engine_available(args.engine):
        print(
            f"Error: engine '{args.engine}' is not installed. "
            f"Install it with: {engine_install_hint(args.engine)}",
            file=sys.stderr,
        )
        return 2

    model = args.model or (
        DEFAULT_WHISPER_MODEL if args.engine == "whisper" else next(iter(engine.models))
    )

    language = None if args.language == "auto" else args.language
    if engine.languages is None:
        language = None  # Parakeet auto-detects its language
    elif language is not None and language not in engine.languages:
        print(
            f"Error: language '{language}' is not supported by {engine.label}. "
            f"Supported: {', '.join(sorted(engine.languages))} (or 'auto').",
            file=sys.stderr,
        )
        return 2

    try:
        ladder = parse_ladder(args.chunk_ladder)
    except ValueError as exc:
        print(f"Error: --chunk-ladder: {exc}", file=sys.stderr)
        return 2

    settings = Settings(
        input_path=args.input,
        start=args.start,
        duration=args.duration,
        engine=args.engine,
        whisper_model=model,
        language=language,
        diarize=not args.no_diarize,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        backend=args.backend,
        beam_size=None if args.beam_size == "auto" else int(args.beam_size),
        stall_timeout=args.stall_timeout,
        chunk_fallback=not args.no_chunk_fallback,
        chunk_ladder=ladder,
        try_whole_file_first=not args.chunk_first,
        skip_failed_chunks=not args.fail_on_stall,
    )
    print(
        f"Backend: {describe_runtime(settings.backend, settings.device, settings.compute_type)}"
        f", beam size {settings.beam_size}"
        + (
            f" | Diarization: pyannote · {settings.diarization_device}"
            if settings.diarize
            else " | Diarization: off"
        ),
        file=sys.stderr,
    )
    if settings.stall_timeout > 0 and settings.chunk_fallback and settings.chunk_ladder:
        print(
            f"Stall recovery: after {settings.stall_timeout:g}s without progress, "
            f"retry in {format_ladder(settings.chunk_ladder)} min chunks",
            file=sys.stderr,
        )
    elif settings.stall_timeout > 0:
        print(
            f"Stall recovery: fail after {settings.stall_timeout:g}s without progress "
            "(chunk fallback off)",
            file=sys.stderr,
        )
    else:
        print("Stall recovery: off (--stall-timeout 0)", file=sys.stderr)

    segments = run_pipeline(settings, progress=lambda m: print(m, file=sys.stderr))
    if not segments:
        print("No speech detected.", file=sys.stderr)
        return 1

    in_path = Path(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = in_path.stem

    for fmt in args.formats:
        out_file = out_dir / f"{stem}.{fmt}"
        out_file.write_text(export(segments, fmt), encoding="utf-8")
        print(f"Wrote {out_file}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
