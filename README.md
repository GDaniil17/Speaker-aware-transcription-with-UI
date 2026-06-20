# 🎙️ Speaker Transcribe

Local, speaker-aware transcription for interviews, podcasts, and meetings.
It combines **Whisper** (faster-whisper) for transcription with **pyannote.audio**
for speaker diarization, so the output tells you *who said what* — ready for
interview annotation, podcast processing, or subtitle generation.

Everything runs **locally**. No audio leaves your machine.

## Features

- Upload an audio/video file **or** point to a local path.
- Speaker diarization + transcription → transcript grouped by speaker.
- Export to **`.txt`**, **`.json`**, **`.srt`**, **`.vtt`**.
- Works on long recordings, with a **fragment mode** (first N seconds / time range)
  for fast iteration and testing.
- Streamlit UI **and** a CLI.
- **Apple Silicon acceleration** via MLX (`mlx-whisper`) — auto-selected on M-series
  Macs for much faster transcription; falls back to faster-whisper (CUDA/CPU)
  elsewhere.
- Reproducible: [uv](https://docs.astral.sh/uv/) for deps, `ruff` for lint/format,
  Docker for a clean environment.

## Prerequisites

1. **HuggingFace token** (needed for diarization):
   - Create one at https://huggingface.co/settings/tokens
   - Accept the model terms at
     https://huggingface.co/pyannote/speaker-diarization-community-1
2. **ffmpeg** — used to decode audio/video. Installed automatically inside Docker.
   Locally: `brew install ffmpeg` (a bundled fallback via `imageio-ffmpeg` is used
   if it's missing).

Copy the token into a `.env` file:

```bash
cp .env.example .env
# then edit .env and set HF_TOKEN=hf_...
```

## Run locally (uv)

With `make` (from zero):

```bash
make run        # uv sync + launch the UI at http://localhost:8501
make cli FILE="sample-interview.mp3" DURATION=60 MODEL=small   # quick fragment test
make check      # lint + tests
make help       # all targets
```

Or directly:

```bash
uv sync                              # create venv + install deps (first run downloads a lot)
uv run streamlit run app.py         # open http://localhost:8501
```

In the UI: pick a Whisper model (start with `small` for quick tests), choose a
fragment (e.g. *First 60 seconds*), provide a file, and click **Transcribe**.

> 💡 The first run also downloads the Whisper and pyannote models. Subsequent
> runs reuse the HuggingFace cache.

### CLI (quick fragment tests)

```bash
# First 60 seconds of the sample interview, fast model, txt + srt output:
uv run transcribe \
  --input "sample-interview.mp3" \
  --duration 60 --model small --formats txt srt

# A specific time range, autodetect language:
uv run transcribe --input sample-meeting.mp4 --start 120 --duration 90 \
  --language auto --formats txt json srt vtt
```

## Run with Docker

```bash
make docker-up           # build + run; open http://localhost:8501
# or:  docker compose up --build
# (HF_TOKEN is read from your shell or the .env file)
```

- Downloaded models are cached in `./hf_cache` (mounted volume), so they persist
  across runs.
- To use the **local path** input inside the container, drop files into `./data`
  and reference them as `/data/some_file.mp3`.

> The Docker image is CPU-only (no GPU passthrough on macOS). On a CUDA host the
> code auto-detects the GPU when torch sees one.

## Development

```bash
uv run ruff check .      # lint
uv run ruff format .     # format
uv run pytest            # tests (export + audio; no model download needed)
```

## Project layout

```
app.py                       # Streamlit UI
src/speaker_transcribe/
  config.py                  # settings + device/token detection
  audio.py                   # ffmpeg: fragment extraction → 16 kHz mono WAV
  pipeline.py                # diarization (pyannote) + transcription (Whisper)
  export.py                  # txt / json / srt / vtt exporters
  cli.py                     # command-line entry point (`transcribe`)
tests/                       # unit tests
code_from_notebook_base.py   # original example (kept for reference)
```

## Notes & troubleshooting

- **`large-v3` on CPU is slow.** Use `small`/`medium` for testing, or the fragment
  mode, and switch to `large-v3` for final quality. `large-v3-turbo` (alias `turbo`)
  is multilingual and much faster than `large-v3` with near-large quality.
- **Model language coverage.** `tiny…large-v3` and `turbo` are multilingual;
  the `distil-*`/`*.en` models are **English-only**. Run `uv run transcribe
  --list-models` to see the full list, or pick one in the UI dropdown.
- **Backends.** `auto` picks **MLX** on Apple Silicon (Mac GPU, fastest there)
  and **faster-whisper** otherwise (CUDA, else CPU). Override in the UI sidebar
  or with `--backend {auto,mlx,faster-whisper}`. MLX downloads `mlx-community/*`
  model weights on first use; `beam size`/`VAD` apply to faster-whisper only.
- **Diarization** runs through pyannote and auto-uses the fastest torch device
  (`cuda` > `mps` Apple GPU > `cpu`), falling back to CPU if an op is
  unsupported. On Apple Silicon, MPS makes it dramatically faster (≈9× in our
  test) — without it, diarizing a long file on CPU can look "stuck" for minutes.
  Progress is reported per pyannote step on the progress bar.
- **Diarization auth errors** usually mean the token is missing or you haven't
  accepted the model terms on HuggingFace.
