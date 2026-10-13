"""The Streamlit page. Streamlit reruns `main` top to bottom on every interaction."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from speaker_transcribe.ui import media
from speaker_transcribe.ui.i18n import remember_locale, tr
from speaker_transcribe.ui.player import render_transcript
from speaker_transcribe.ui.run import run_with_progress, show_last_run, show_token_needed
from speaker_transcribe.ui.sidebar import Choices, render_sidebar
from speaker_transcribe.ui.source import Source, render_source

# Set by the quick-start launchers: Whisper only, no speaker identification.
BASIC_MODE_ENV = "SPEAKER_TRANSCRIBE_BASIC"


def main(app_dir: Path) -> None:
    """Draw the page. `app_dir` holds the entry script and the served `static/`."""
    basic_mode = os.environ.get(BASIC_MODE_ENV) == "1"
    st.set_page_config(page_title="Speaker Transcribe", page_icon="🎙️", layout="wide")
    media.set_static_dir(app_dir / "static")
    media.clear_static_previews_once()

    st.title("🎙️ " + tr("Speaker-aware transcription"))
    st.caption(tr("Transcribe locally and identify speakers. Your files stay on this machine."))
    if basic_mode:
        st.info(tr("Upload a file, then start transcription. Download the model first."))

    choices = render_sidebar(basic_mode)
    source = render_source()

    run = st.button(
        f"▶ {tr('Transcribe')}",
        type="primary",
        use_container_width=True,
        disabled=not source.ready or not choices.can_run,
    )
    if source.ready and choices.engine_ok and not choices.model_ready:
        st.caption(tr("Download the selected model in Settings to start processing."))

    if not run:
        show_last_run()
    elif choices.needs_token:
        show_token_needed()
    else:
        _transcribe(source, choices)

    if st.session_state.get("segments"):
        render_transcript(st.session_state.segments)

    remember_locale()


def _transcribe(source: Source, choices: Choices) -> None:
    """Run the pipeline on the chosen recording and keep the result in the session."""
    cleanup: Path | None = None
    if source.uploaded is not None:
        upload = source.uploaded
        suffix = Path(upload.name).suffix or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(upload.getbuffer())
        input_path = tmp.name
        cleanup = Path(tmp.name)
    else:
        input_path = source.path
    try:
        segments = run_with_progress(choices.settings_for(input_path))
    finally:
        if cleanup is not None:
            cleanup.unlink(missing_ok=True)

    if segments is None:
        st.session_state.pop("segments", None)
        return
    st.session_state.segments = segments
    if source.uploaded is not None:
        media.remember_media("bytes", bytes(source.uploaded.getbuffer()), source.uploaded.name)
    else:
        media.remember_media("path", input_path, Path(input_path).name)
    if not segments:
        st.warning(tr("No speech detected in the selected fragment."))
