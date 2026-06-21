"""Command-line entry point — useful for quick tests on short fragments.

Example:
    uv run transcribe --input sample-interview.mp3 --duration 60 --model small
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from speaker_transcribe.config import (
    ENGINES,
    Settings,
    describe_runtime,
    engine_available,
    engine_install_hint,
)
from speaker_transcribe.export import EXPORTERS, export
from speaker_transcribe.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transcribe",
        description="Speaker-aware transcription (Whisper + pyannote diarization).",
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
        "qwen3-asr (Qwen3-ASR-1.7B, 30 langs). Non-Whisper engines need an extra "
        "install — see --list-models.",
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
        default="ru",
        help="Language code, or 'auto' to autodetect.",
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

    model = args.model or next(iter(engine.models))

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

    settings = Settings(
        input_path=args.input,
        start=args.start,
        duration=args.duration,
        engine=args.engine,
        whisper_model=model,
        language=language,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        backend=args.backend,
        beam_size=None if args.beam_size == "auto" else int(args.beam_size),
    )
    print(
        f"Backend: {describe_runtime(settings.backend, settings.device, settings.compute_type)}"
        f", beam size {settings.beam_size}"
        f" | Diarization: pyannote · {settings.diarization_device}",
        file=sys.stderr,
    )

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
