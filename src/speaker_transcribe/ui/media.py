"""The recording shown next to the transcript: inline, or served from ./static.

Media up to 150 MB is inlined as a `data:` URI. Larger files are exposed under
the app's `static/` folder, which Streamlit serves at `app/static/...` when
`server.enableStaticServing` is on (see .streamlit/config.toml). Video that a
browser may not decode also gets an MP3 copy of its audio as a fallback.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from speaker_transcribe.audio import get_ffmpeg
from speaker_transcribe.ui.i18n import tr

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
MEDIA_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"} | VIDEO_EXTS

_MIME_BY_EXT = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".aac": "audio/aac",
    ".mp4": "video/mp4",
    ".mov": "video/mp4",
    ".m4v": "video/mp4",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
}

MAX_INLINE_BYTES = 150 * 1024 * 1024
MAX_STATIC_PREVIEW_BYTES = 950 * 1024 * 1024

# Video every current browser decodes; anything else gets an MP3 fallback.
_UNIVERSAL_VIDEO_CODECS = {"h264", "vp8", "vp9", "av1"}
_UNIVERSAL_VIDEO_EXTS = {".mp4", ".m4v", ".webm", ".mov"}

# Streamlit serves the `static/` folder next to the app's entry script. The
# entry script sets this at start-up (see `ui.app.main`).
STATIC_DIR = Path.cwd() / "static"


def set_static_dir(path: Path) -> None:
    global STATIC_DIR
    STATIC_DIR = path


@dataclass(frozen=True)
class Media:
    """The recording of the last run: uploaded bytes or a local file."""

    kind: str  # "bytes" or "path"
    payload: bytes | str
    name: str
    size: int

    @property
    def ext(self) -> str:
        return Path(self.name).suffix.lower()

    @property
    def mime(self) -> str:
        return _MIME_BY_EXT.get(self.ext, "application/octet-stream")

    @property
    def is_video(self) -> bool:
        return self.ext in VIDEO_EXTS


def remember_media(kind: str, payload: bytes | str, name: str) -> None:
    """Keep the recording of a finished run for the player."""
    st.session_state.media_source = (kind, payload, name)


def current_media() -> Media | None:
    """The recording of the last run, or None when there is none (or it is gone)."""
    src = st.session_state.get("media_source")
    if not src:
        return None
    kind, payload, name = src
    if kind == "bytes":
        return Media(kind, payload, name, len(payload))
    try:
        return Media(kind, payload, name, Path(payload).stat().st_size)
    except OSError:
        return None


def clear_static_previews(keep: str = "") -> None:
    """Drop served preview files so the static dir doesn't grow.

    Files of the same media as `keep` (e.g. its audio-only copy) stay.
    """
    if not STATIC_DIR.is_dir():
        return
    keep_stem = keep.split(".", 1)[0]
    for f in STATIC_DIR.glob("preview_*"):
        if not keep or f.name.split(".", 1)[0] != keep_stem:
            f.unlink(missing_ok=True)


@st.cache_resource
def clear_static_previews_once() -> bool:
    """Clear old previews once per server start.

    Streamlit re-runs the script on every interaction; clearing each time made
    every click rewrite the media preview (and redo its MP3 copy).
    """
    clear_static_previews()
    return True


def _preview_key(media: Media) -> str:
    """Content-keyed id for the media's preview files."""
    if media.kind == "bytes":
        ident = f"{media.name}:{media.size}"
    else:
        p = Path(media.payload)
        stat = p.stat()
        ident = f"{p.resolve()}:{stat.st_size}:{int(stat.st_mtime)}"
    return hashlib.sha1(ident.encode()).hexdigest()[:16]


def _static_media_url(media: Media) -> str | None:
    """Expose the media under ./static and return its `app/static/...` URL.

    Uploads are written out; local paths are hard-linked (else copied). Idempotent
    via a content-keyed filename. Returns None on failure.
    """
    try:
        STATIC_DIR.mkdir(parents=True, exist_ok=True)
        target = STATIC_DIR / f"preview_{_preview_key(media)}{media.ext or '.bin'}"
        clear_static_previews(keep=target.name)
        if not target.exists():
            if media.kind == "bytes":
                target.write_bytes(media.payload)
            else:
                try:
                    os.link(media.payload, target)
                except OSError:
                    shutil.copy2(media.payload, target)
        return f"app/static/{target.name}"
    except OSError:
        return None


@st.cache_resource
def static_serving_works() -> bool:
    """Whether the running server really serves ./static at app/static/.

    `server.enableStaticServing` is applied once, at server start, but
    st.get_option() reflects later edits to .streamlit/config.toml. A server
    started before the option was on answers app/static/... with the app's HTML
    page, which leaves the player empty, so ask the server itself.
    """
    if not st.get_option("server.enableStaticServing"):
        return False
    probe = STATIC_DIR / "preview_probe.txt"
    base = (st.get_option("server.baseUrlPath") or "").strip("/")
    url = f"http://127.0.0.1:{st.get_option('server.port')}/{base + '/' if base else ''}"
    try:
        STATIC_DIR.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok")
        with urllib.request.urlopen(f"{url}app/static/{probe.name}", timeout=3) as resp:
            return resp.read(16) == b"ok"
    except OSError:
        # Bound to an address other than localhost: trust the option.
        return True
    finally:
        probe.unlink(missing_ok=True)


def _video_codec(path: Path) -> str | None:
    """Codec of the first video stream, read from `ffmpeg -i` (no ffprobe needed)."""
    try:
        out = subprocess.run(
            [get_ffmpeg(), "-hide_banner", "-nostdin", "-i", str(path)],
            capture_output=True,
            text=True,
            timeout=60,
        ).stderr
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return None
    match = re.search(r"Stream #\S+.*?: Video: (\w+)", out)
    return match.group(1) if match else None


@st.cache_resource
def _mp3_jobs() -> dict:
    """MP3 previews being encoded, shared across reruns and sessions."""
    return {"lock": threading.Lock(), "running": set()}


def _encode_mp3(source: Path | bytes, ext: str, target: Path) -> None:
    """Encode the audio track to `target` (written whole, then renamed)."""
    part = target.with_name(target.name.replace(".mp3", ".part.mp3"))
    try:
        with tempfile.TemporaryDirectory(prefix="preview_") as tmp:
            if isinstance(source, bytes):
                src_path = Path(tmp) / f"in{ext}"
                src_path.write_bytes(source)
            else:
                src_path = source
            cmd = [
                get_ffmpeg(),
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(src_path),
                "-vn",
                "-ac",
                "1",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "96k",
                str(part),
            ]
            if subprocess.run(cmd, capture_output=True).returncode == 0:
                part.replace(target)
    except (OSError, RuntimeError):
        pass
    finally:
        part.unlink(missing_ok=True)


def mp3_fallback_url(media: Media) -> str | None:
    """URL of an MP3 copy of the audio, for video browsers may not decode.

    H.264/VP8/VP9/AV1 in mp4/webm/mov play everywhere and get none. Anything
    else (HEVC from iPhones and macOS screen recordings, mkv, avi, ...) gets an
    MP3 encoded in the background (~10 s per hour of audio); the player switches
    to it if the browser reports a decode error. Needs static file serving.
    """
    if not static_serving_works():
        return None
    try:
        key = _preview_key(media)
        target = STATIC_DIR / f"preview_{key}.mp3"
        url = f"app/static/{target.name}"
        clear_static_previews(keep=target.name)  # other media's previews
        if target.exists():
            return url
        if media.kind == "bytes":
            # A directory rather than NamedTemporaryFile: Windows will not let
            # ffmpeg open a file this process still holds open.
            with tempfile.TemporaryDirectory(prefix="probe_") as tmp:
                probe = Path(tmp) / f"probe{media.ext}"
                probe.write_bytes(media.payload[: 4 * 1024 * 1024])  # the header is enough
                codec = _video_codec(probe)
            source: Path | bytes = media.payload
        else:
            codec = _video_codec(Path(media.payload))
            source = Path(media.payload)
    except OSError:
        return None
    if media.ext in _UNIVERSAL_VIDEO_EXTS and codec in _UNIVERSAL_VIDEO_CODECS:
        return None

    jobs = _mp3_jobs()
    with jobs["lock"]:
        if key in jobs["running"]:
            return url
        jobs["running"].add(key)

    def work() -> None:
        try:
            STATIC_DIR.mkdir(parents=True, exist_ok=True)
            _encode_mp3(source, media.ext, target)
        finally:
            with jobs["lock"]:
                jobs["running"].discard(key)

    threading.Thread(target=work, name=f"mp3-preview-{key}", daemon=True).start()
    return url


def player_src(media: Media) -> str | None:
    """`data:` URI when small enough to inline, else an `app/static/...` URL."""
    if media.size <= MAX_INLINE_BYTES:
        try:
            data = media.payload if media.kind == "bytes" else Path(media.payload).read_bytes()
        except OSError:
            data = None
        if data is not None:
            return f"data:{media.mime};base64,{base64.b64encode(data).decode('ascii')}"
    if media.size > MAX_STATIC_PREVIEW_BYTES:
        return None
    if not static_serving_works():
        # app/static/... would answer with the app's HTML page, not the media.
        st.warning(
            tr(
                "No preview for media over 150 MB: this server doesn't serve static files. "
                "Restart the app so it picks up enableStaticServing from .streamlit/config.toml."
            )
        )
        return None
    return _static_media_url(media)
