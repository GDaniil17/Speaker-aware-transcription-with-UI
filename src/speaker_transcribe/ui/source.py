"""Choosing the recording: an upload or a file on this machine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from speaker_transcribe.ui.i18n import locale_key, tr
from speaker_transcribe.ui.media import MEDIA_EXTS

UPLOAD_TYPES = sorted(ext.lstrip(".") for ext in MEDIA_EXTS)


@dataclass
class Source:
    """The recording picked in the main panel, if any."""

    uploaded: object | None = None  # streamlit UploadedFile
    path: str | None = None

    @property
    def ready(self) -> bool:
        return self.uploaded is not None or self.path is not None


def render_source() -> Source:
    st.subheader(tr("Add a recording"))
    modes = ["Upload file", "Local path"]
    mode = st.segmented_control(
        tr("Source"),
        modes,
        default="Upload file",
        format_func=tr,
        label_visibility="collapsed",
        key=locale_key("input_source", modes),
    )
    if mode == "Upload file":
        uploaded = st.file_uploader(
            tr("Audio or video file"),
            type=UPLOAD_TYPES,
            label_visibility="collapsed",
            key="uploaded_media",
        )
        return Source(uploaded=uploaded)  # the uploader itself shows name and size

    # A file chosen in the browser below lands here on the next run: a widget's
    # value cannot change once the widget has been drawn in the current run.
    if "_browsed_path" in st.session_state:
        st.session_state.picked_path = st.session_state.pop("_browsed_path")
    if "picked_path" not in st.session_state:
        st.session_state.picked_path = ""
    typed = st.text_input(
        tr("Path to an existing file on this machine"),
        key="picked_path",
        placeholder="/path/to/recording.mp3",
    )
    # In Docker, recordings are mounted at /data.
    default_root = "/data" if Path("/data").is_dir() else str(Path.home())
    with st.expander(f"📂 {tr('Browse files on this machine')}"):
        _file_browser(default_root)
    if not typed:
        return Source()
    selected = Path(typed).expanduser()
    if not selected.is_file():
        st.error(tr("File not found."))
        return Source()
    st.caption(f"✓ {selected.name}")
    return Source(path=str(selected))


def _file_browser(start_dir: str) -> None:
    """Server-side folder/file navigator; writes the choice to picked_path."""
    ss = st.session_state
    cur = Path(ss.get("browse_dir", start_dir)).expanduser()
    if not cur.is_dir():
        cur = Path(start_dir).expanduser()
    ss.browse_dir = str(cur)

    st.caption(f"📂 `{cur}`")

    try:
        entries = sorted(cur.iterdir(), key=lambda p: p.name.lower())
    except PermissionError:
        st.error(tr("Permission denied for this folder."))
        return

    dirs = [p for p in entries if p.is_dir() and not p.name.startswith(".")]
    files = [p for p in entries if p.is_file() and p.suffix.lower() in MEDIA_EXTS]

    placeholder = tr("— select a folder or file —")
    up = tr("⬆️  .. (up one level)")
    options = [placeholder, up]
    options += [f"📁 {p.name}" for p in dirs]
    options += [f"🎵 {p.name}" for p in files]

    choice = st.selectbox(tr("Browse"), options, key=f"browse_{cur}")

    if choice == up:
        ss.browse_dir = str(cur.parent)
        st.rerun()
    elif choice.startswith("📁 "):
        ss.browse_dir = str(cur / choice[2:].strip())
        st.rerun()
    elif choice.startswith("🎵 "):
        picked = str(cur / choice[2:].strip())
        if picked != ss.get("_last_browsed"):  # a new pick, not the same one again
            ss._last_browsed = ss._browsed_path = picked
            st.rerun()
