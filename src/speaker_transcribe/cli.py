"""Command-line entry point — useful for quick tests on short fragments.

Example:
    uv run transcribe --input sample-interview.mp3 --duration 60 --model small
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from speaker_transcribe.config import WHISPER_MODELS, Settings, describe_runtime
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
        "--model",
        default="large-v3",
        metavar="NAME",
        help="Whisper model. Multilingual: tiny, base, small, medium, large-v3, "
        "large-v3-turbo, turbo. English-only: distil-large-v3, distil-large-v3.5, "
        "distil-medium.en, distil-small.en. See --list-models.",
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        help="List known Whisper models with language coverage and exit.",
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
        width = max(len(m) for m in WHISPER_MODELS)
        for name, desc in WHISPER_MODELS.items():
            print(f"  {name:<{width}}  {desc}")
        return 0

    if not args.input:
        print("Error: --input is required (or use --list-models).", file=sys.stderr)
        return 2

    settings = Settings(
        input_path=args.input,
        start=args.start,
        duration=args.duration,
        whisper_model=args.model,
        language=None if args.language == "auto" else args.language,
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
