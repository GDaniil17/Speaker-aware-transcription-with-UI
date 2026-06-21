# Speaker Transcribe

Local transcription with speaker labels.

Speaker Transcribe uses Whisper (or an alternative local ASR engine) for
speech-to-text and pyannote.audio for speaker diarization. It can process audio
or video files and export transcripts as text, JSON, SRT, or VTT.

## What It Does

- Transcribes audio and video files.
- Detects speakers and groups transcript lines by speaker.
- Lets you process the whole file, the first N seconds, or a time range.
- Runs from a Streamlit UI or the `transcribe` CLI.
- Exports `.txt`, `.json`, `.srt`, and `.vtt`.
- Runs locally. Audio is not sent to an external transcription API.

## Requirements

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/)
- ffmpeg
- HuggingFace token for diarization

For diarization, create a HuggingFace token and accept the pyannote model terms:

- https://huggingface.co/settings/tokens
- https://huggingface.co/pyannote/speaker-diarization-community-1

Create `.env`:

```bash
cp .env.example .env
```

Then set:

```bash
HF_TOKEN=hf_...
```

If ffmpeg is missing:

```bash
brew install ffmpeg
```

Docker installs ffmpeg inside the image.

## Quick Start

```bash
make run
```

Open:

```text
http://localhost:8501
```

In the UI:

1. Choose a Whisper model.
2. Choose a language, or use auto-detect.
3. Choose whole file or a fragment.
4. Upload a file or enter a local path.
5. Click `Transcribe`.
6. Download the formats you need.

Use `small` for quick tests. Use `large-v3` or `large-v3-turbo` for better final output.

## CLI

Process the first 60 seconds:

```bash
uv run transcribe \
  --input sample-interview.mp3 \
  --duration 60 \
  --model small \
  --language auto \
  --formats txt srt
```

Process a specific range:

```bash
uv run transcribe \
  --input sample-meeting.mp4 \
  --start 120 \
  --duration 90 \
  --language auto \
  --formats txt json srt vtt
```

Write output files to a custom directory:

```bash
uv run transcribe \
  --input sample-meeting.mp4 \
  --out-dir outputs \
  --formats txt json
```

List available Whisper models:

```bash
uv run transcribe --list-models
```

## Docker

```bash
make docker-up
```

Open:

```text
http://localhost:8501
```

Docker notes:

- `HF_TOKEN` is read from your shell or `.env`.
- Model downloads are cached in `./hf_cache`.
- Put files in `./data` when using local paths inside Docker.
- In Docker, refer to those files as `/data/file.mp3`.

## Models And Backends

Backend selection:

- `auto`: MLX on Apple Silicon, otherwise faster-whisper.
- `mlx`: Apple Silicon GPU backend.
- `faster-whisper`: CUDA if available, otherwise CPU.

Useful models:

| Model | Best for |
| --- | --- |
| `small` | Fast tests |
| `medium` | Better quality without huge runtime |
| `large-v3` | Best quality, slow on CPU |
| `large-v3-turbo` / `turbo` | Near-large quality, faster |
| `distil-*` / `*.en` | English-only, faster |

## Other ASR Engines

Besides Whisper, you can transcribe with other local models. They all run
locally, produce timestamps, and work with diarization. Pick the engine in the
sidebar (UI) or with `--engine` (CLI). Each needs its own optional install:

| Engine | Model | Languages | Install |
| --- | --- | --- | --- |
| `whisper` | Whisper (multiple sizes) | ~99, multilingual | included |
| `parakeet` | `parakeet-tdt-0.6b-v3` (NVIDIA) | 25 European, auto-detected | `make install-parakeet` (or `make install-parakeet-mlx` on Apple Silicon) |
| `qwen3-asr` | `Qwen3-ASR-1.7B` (Qwen) | 30, timestamps via forced aligner | `make install-qwen3-asr` |
| `sensevoice` | `SenseVoiceSmall` (FunAudioLLM) | 5 (zh, yue, en, ja, ko) | `make install-sensevoice` |

Notes:

- Parakeet auto-detects the language; the language selector is disabled for it.
- On Apple Silicon, Parakeet uses the MLX backend (`parakeet-mlx`); elsewhere it
  uses NeMo.
- Qwen3-ASR also downloads `Qwen3-ForcedAligner-0.6B` to produce timestamps.
- SenseVoice also downloads an `fsmn-vad` model for segmenting long audio.
- These engines pull heavy, version-pinned dependencies that can conflict with
  each other or with the pinned torch/pyannote — if resolution fails, install the
  engine in a dedicated virtual environment.

Install and run every engine at once:

```bash
make install-all   # install all engines (one uv sync)
make run-all       # install all engines, then launch the UI
```

`make run-all` puts the engines and the launch into a single `uv sync` —
installing separately and then running `make run` would re-sync without the
extras and remove them. To run with a specific subset instead, pass `EXTRAS`:

```bash
make run EXTRAS="parakeet-mlx qwen3-asr sensevoice"
```

Both auto-pick the `parakeet-mlx` backend on Apple Silicon and `parakeet` (NeMo)
elsewhere. If the combined install fails to resolve, install one engine per venv.

### Choosing a Model for Your Hardware

The UI shows a hardware-aware **"Which engine & model should I use?"** panel in the
sidebar (it auto-detects your device). The same guidance:

| Hardware | Recommended picks | Why |
| --- | --- | --- |
| **NVIDIA GPU (CUDA)** | Whisper `large-v3` / `large-v3-turbo`; Parakeet; Qwen3-ASR | float16 keeps large models fast; Qwen3-ASR gives top accuracy when you have VRAM. |
| **Apple Silicon (MLX / MPS)** | Whisper `large-v3-turbo` (MLX); Parakeet (MLX); SenseVoice | MLX runs on the Apple GPU; `parakeet-mlx` avoids heavy NeMo. |
| **CPU only** | Whisper `small` / `medium`; `distil-*` for English; SenseVoice | int8 keeps it usable; avoid `large-v3` (slow) and GPU-hungry Qwen3-ASR. |

Language support by engine:

| Engine | Languages |
| --- | --- |
| `whisper` | 99 languages (multilingual); `distil-*` variants are English-only |
| `parakeet` | 25 European languages, auto-detected |
| `qwen3-asr` | 30 languages |
| `sensevoice` | 5 languages: Chinese, Cantonese, English, Japanese, Korean |

Rule of thumb: Whisper for the widest language coverage, Parakeet for fast
European-language audio, Qwen3-ASR for top accuracy on a GPU, and SenseVoice when
your audio is Chinese/Cantonese/English/Japanese/Korean.

List engines and their models:

```bash
uv run transcribe --list-models
```

Transcribe with an alternative engine:

```bash
uv run transcribe --engine parakeet --input sample.mp3 --duration 60
uv run transcribe --engine qwen3-asr --language en --input sample.mp3 --duration 60
```

## Output Files

The CLI writes files next to the input unless `--out-dir` is set.

Supported formats:

- `txt`: readable transcript
- `json`: structured segments
- `srt`: subtitles
- `vtt`: web subtitles

## Development

```bash
make test
make lint
make check
```

Direct commands:

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```

## Project Layout

```text
app.py                       Streamlit UI
src/speaker_transcribe/
  audio.py                   audio extraction and fragment handling
  cli.py                     transcribe command
  config.py                  settings, models, device detection
  export.py                  txt/json/srt/vtt exporters
  pipeline.py                ASR engines (Whisper/Parakeet/Qwen3) + pyannote pipeline
tests/                       unit tests
```

## Screenshots

<p>
  <img src="docs/screenshots/ui-desktop.png" alt="Speaker Transcribe desktop UI with settings sidebar and upload workflow" width="100%">
</p>

<p>
  <img src="docs/screenshots/processed-file.png" alt="Processed Movie.mp4 with transcript, video preview, and speaker-labeled lines" width="100%">
</p>

<p>
  <img src="docs/screenshots/speaker-editor.png" alt="Speaker editor with renamed speaker label and editable speaker controls" width="100%">
</p>

## Troubleshooting

`HF_TOKEN` errors:

- Check `.env`.
- Accept the pyannote model terms on HuggingFace.

Slow transcription:

- Use `small` or `medium` while testing.
- Use a fragment with `--duration`.
- Avoid `large-v3` on CPU for quick runs.

No speaker labels:

- Make sure diarization is enabled.
- Check that `HF_TOKEN` is valid.
- Set realistic `--min-speakers` and `--max-speakers`.

Docker local paths do not work:

- Put files in `./data`.
- Use `/data/<filename>` in the app.
