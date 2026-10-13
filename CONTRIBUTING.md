# Contributing

Thanks for helping! Bug reports, fixes, new engines and translations are all welcome.

## Development setup

You need [uv](https://docs.astral.sh/uv/). It installs the right Python by itself.

```bash
git clone https://github.com/GDaniil17/Speaker-aware-transcription-with-UI.git
cd Speaker-aware-transcription-with-UI
uv sync                      # .venv with every core dependency + dev tools
uv run streamlit run app.py  # the UI
uv run transcribe --help     # the CLI
```

Before opening a pull request:

```bash
uv run ruff check .
uv run ruff format .
uv run pytest
```

(or `make check` on macOS/Linux). CI runs the same checks on Linux, macOS and Windows.
The tests need no model downloads and no GPU.

## How the code is organised

```text
app.py                          Streamlit entry point (a few lines; calls ui.app.main)
run.sh / run.ps1 / run.bat      One-command launchers (quick start and --full)
requirements/                   Light dependency sets for the quick-start launchers
src/speaker_transcribe/
  catalog.py                    Engines, models, languages, Hugging Face repos — pure data
  runtime.py                    Device/backend detection, which engines are installed
  config.py                     Settings for one run; .env loading
  hub.py                        Hugging Face cache checks and downloads with progress
  audio.py                      ffmpeg conversion to 16 kHz mono WAV, slicing, samples
  diarization.py                pyannote speaker diarization
  engines/                      One module per ASR engine + `transcribe()` dispatcher
  transcript.py                 TranscriptSegment and speaker attribution helpers
  chunking.py                   Stall recovery: retry stuck audio in smaller chunks
  worker.py                     Runs transcription in a killable child process
  pipeline.py                   End to end: audio → diarization → transcription
  export.py                     txt / json / srt / vtt
  model_downloads.py            Background model downloads for the UI
  engine_install.py             Installing optional engines from the UI
  cli.py                        The `transcribe` command
  ui/                           Streamlit interface
    app.py                      Page layout
    sidebar.py                  Settings sidebar → `Choices`
    source.py                   Upload / local file picker
    run.py                      Live progress while the pipeline runs
    player.py, assets/          Media player + editable transcript (HTML/JS component)
    media.py                    Serving large media and MP3 fallbacks
    i18n.py                     English/Russian interface strings
tests/                          Unit tests (pytest)
```

## Adding an ASR engine

1. Add the engine, its models and languages to `catalog.py` (`ENGINES`).
2. Write `engines/<name>.py` with a `transcribe(wav_path, settings, diar, sp_map,
   progress, on_progress)` generator that yields `TranscriptSegment`s. Import the
   engine's library inside the function so it stays optional, and use
   `transcript.emit_segment` / `split_by_speaker` for speaker attribution.
3. Route to it in `engines/__init__.py`, report availability in
   `runtime.engine_available`, and list its Hugging Face repos in
   `model_downloads.model_repos`.
4. If it needs extra packages, add an optional dependency group in
   `pyproject.toml` (`[project.optional-dependencies]`), then `uv lock`.

## Translating the interface

All interface text goes through `tr("English text")`. Russian lives in
`ui/i18n.py` (`RU`). To fix a translation, edit that dictionary. Adding a third
language means adding a dictionary and an entry in the EN/RU selector in
`ui/sidebar.py`.

## Screenshots

The README screenshots use a synthetic two-voice dialogue (macOS `say`). If you
change the UI noticeably, please refresh them in `docs/screenshots/`.
