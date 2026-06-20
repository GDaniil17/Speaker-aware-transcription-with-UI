"""Minimal Streamlit UI for speaker-aware transcription.

Run with:  uv run streamlit run app.py
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import tempfile
from html import escape
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from speaker_transcribe.config import (
    WHISPER_LANGUAGES,
    WHISPER_MODELS,
    Settings,
    auto_beam_size,
    describe_runtime,
    detect_backend,
    detect_device,
    detect_diarization_device,
    downloaded_whisper_models,
    get_hf_token,
    is_english_only_model,
    mlx_available,
)
from speaker_transcribe.export import EXPORTERS, export
from speaker_transcribe.pipeline import TranscriptSegment, run_pipeline

st.set_page_config(page_title="Speaker Transcribe", page_icon="🎙️", layout="wide")
st.title("🎙️ Speaker-aware transcription")
st.caption("Whisper transcription + pyannote diarization — runs fully locally.")

MODEL_NAMES = list(WHISPER_MODELS)


def format_eta(seconds: float) -> str:
    """Human-readable remaining time, e.g. '1m 05s' or '12s'."""
    seconds = max(0, int(round(seconds)))
    m, s = divmod(seconds, 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def format_clock(seconds: float) -> str:
    """Format a timestamp in seconds as clock time, e.g. '28:18.8' or '1:05:03.2'."""
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:04.1f}" if h else f"{m:02d}:{s:04.1f}"


SPEAKER_PALETTE = [
    "#4363d8", "#3cb44b", "#e6194B", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#9A6324",
]


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

MEDIA_EXTS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac",
} | VIDEO_EXTS

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

STATIC_DIR = Path(__file__).resolve().parent / "static"


def media_meta(name: str) -> tuple[str, bool]:
    """Return (mime_type, is_video) for a media filename."""
    ext = Path(name).suffix.lower()
    return _MIME_BY_EXT.get(ext, "application/octet-stream"), ext in VIDEO_EXTS


def _media_info() -> tuple[str, object, str, str, bool, int] | None:
    """(kind, payload, name, mime, is_video, size) for the current input, or None."""
    src = st.session_state.get("media_source")
    if not src:
        return None
    kind, payload, name = src
    mime, is_video = media_meta(name)
    if kind == "bytes":
        return kind, payload, name, mime, is_video, len(payload)
    p = Path(payload)
    if not p.is_file():
        return None
    try:
        size = p.stat().st_size
    except OSError:
        return None
    return kind, payload, name, mime, is_video, size


def _read_media_bytes(info: tuple) -> bytes | None:
    """Full bytes for the current media (only call for small/inline files)."""
    kind, payload, *_ = info
    if kind == "bytes":
        return payload
    try:
        return Path(payload).read_bytes()
    except OSError:
        return None


def _purge_static_previews(keep: str = "") -> None:
    """Drop previously served preview files so the static dir doesn't grow."""
    if not STATIC_DIR.is_dir():
        return
    for f in STATIC_DIR.glob("preview_*"):
        if f.name != keep:
            f.unlink(missing_ok=True)


def _static_media_url(info: tuple) -> str | None:
    """Expose the current media under ./static and return its `app/static/...` URL.

    Uploads are written out; local paths are hard-linked (else copied). Idempotent
    via a content-keyed filename. Returns None on failure.
    """
    kind, payload, name, _mime, _is_video, size = info
    ext = Path(name).suffix.lower() or ".bin"
    try:
        STATIC_DIR.mkdir(parents=True, exist_ok=True)
        if kind == "bytes":
            key = hashlib.sha1(f"{name}:{size}".encode()).hexdigest()[:16]
        else:
            p = Path(payload)
            stt = p.stat()
            key = hashlib.sha1(
                f"{p.resolve()}:{stt.st_size}:{int(stt.st_mtime)}".encode()
            ).hexdigest()[:16]
        target = STATIC_DIR / f"preview_{key}{ext}"
        if not target.exists():
            _purge_static_previews(keep=target.name)
            if kind == "bytes":
                target.write_bytes(payload)
            else:
                try:
                    os.link(payload, target)
                except OSError:
                    shutil.copy2(payload, target)
        return f"app/static/{target.name}"
    except OSError:
        return None


_PLAYER_CSS = """
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, sans-serif;
         color:#e6e6e6; background:transparent; }
  #stage { position:relative; }
  #player { max-height:46vh !important; }
  #caption { position:absolute; left:0; right:0; bottom:52px; text-align:center;
             pointer-events:none; padding:0 8%; }
  #caption span { background:rgba(0,0,0,.72); color:#fff; padding:4px 10px;
                  border-radius:6px; font-size:1.15rem; line-height:1.7;
                  box-decoration-break:clone; -webkit-box-decoration-break:clone; }
  #speakers { display:flex; flex-wrap:wrap; gap:8px; align-items:center;
              margin:12px 0 8px; }
  #speakers > .lbl { opacity:.7; font-size:.85rem; }
  .spkchip { display:flex; align-items:center; gap:6px; padding:3px 6px;
             border:1px solid rgba(128,128,128,.3); border-radius:8px; }
  .spkchip input[type=color] { width:22px; height:22px; border:none; background:none;
                               padding:0; cursor:pointer; }
  .spkchip input[type=text] { background:transparent; border:none; color:inherit;
                              font:inherit; font-weight:700; width:10ch; }
  .spkchip input[type=text]:focus { outline:1px solid rgba(128,128,128,.5);
                                     border-radius:4px; }
  #addspk { cursor:pointer; padding:3px 8px; border:1px dashed rgba(128,128,128,.5);
            border-radius:8px; background:none; color:inherit; font:inherit; }
  #transcript { max-height:300px; overflow-y:auto;
                border:1px solid rgba(128,128,128,.3); border-radius:8px; padding:6px; }
  .line { display:flex; gap:8px; padding:3px 6px; border-radius:6px;
          align-items:flex-start; }
  .line:hover { background:rgba(128,128,128,.12); }
  .line.active { background:rgba(255,214,10,.22); }
  .ts { font-family: ui-monospace, monospace; font-size:.82rem; opacity:.75;
        white-space:nowrap; padding-top:3px; cursor:pointer;
        text-decoration:underline dotted; }
  .ts:hover { opacity:1; color:#ffd60a; }
  .spk { font:inherit; font-size:.78rem; font-weight:700; border:none;
         border-radius:6px; padding:2px 4px; cursor:pointer; color:#fff;
         max-width:15ch; }
  .tx { flex:1; padding-top:2px; }
  #exports { display:flex; gap:8px; margin-top:12px; align-items:center; }
  #exports label { display:flex; align-items:center; gap:4px; cursor:pointer;
                   font-size:.85rem; opacity:.85; margin-right:4px; }
  #exports button { cursor:pointer; padding:6px 12px; font:inherit;
                    border:1px solid rgba(128,128,128,.4); border-radius:8px;
                    background:rgba(128,128,128,.12); color:inherit; }
  #exports button:hover { background:rgba(128,128,128,.25); }
"""

_PLAYER_JS = r"""
  const player = document.getElementById('player');
  const list = document.getElementById('transcript');
  const caption = document.getElementById('caption');
  const spkBar = document.getElementById('speakers');
  let spkSeq = SPEAKERS.length;
  let rows = [];
  let activeIdx = -1;

  function getSpk(key) {
    for (const s of SPEAKERS) if (s.key === key) return s;
    return { key: key, name: key, color: '#888888' };
  }
  function pad(n, l) { return String(n).padStart(l, '0'); }
  function fmt(t) {
    t = Math.max(0, t);
    const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = t % 60;
    const ss = s.toFixed(1).padStart(4, '0');
    return h ? h + ':' + pad(m, 2) + ':' + ss : pad(m, 2) + ':' + ss;
  }
  function tsCode(t, sep) {
    t = Math.max(0, t);
    let h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = Math.floor(t % 60);
    let ms = Math.round((t - Math.floor(t)) * 1000);
    if (ms === 1000) { s += 1; ms = 0; }
    return pad(h, 2) + ':' + pad(m, 2) + ':' + pad(s, 2) + sep + pad(ms, 3);
  }
  function seek(t) {
    player.currentTime = t;
    const p = player.play();
    if (p && p.catch) p.catch(() => {});
  }

  function newSpeaker(name) {
    spkSeq += 1;
    const sp = {
      key: 'spk_' + spkSeq, name: name,
      color: PALETTE[SPEAKERS.length % PALETTE.length],
    };
    SPEAKERS.push(sp);
    return sp;
  }

  function renderSpeakerBar() {
    spkBar.innerHTML = '';
    const lbl = document.createElement('span');
    lbl.className = 'lbl';
    lbl.textContent = 'Speakers:';
    spkBar.appendChild(lbl);
    SPEAKERS.forEach((sp) => {
      const chip = document.createElement('span');
      chip.className = 'spkchip';
      const col = document.createElement('input');
      col.type = 'color';
      col.value = sp.color;
      const nm = document.createElement('input');
      nm.type = 'text';
      nm.value = sp.name;
      nm.style.color = sp.color;
      col.addEventListener('input', () => {
        sp.color = col.value;
        nm.style.color = sp.color;
        renderTranscript();
      });
      nm.addEventListener('input', () => {
        sp.name = nm.value || sp.key;
        renderTranscript();
      });
      chip.appendChild(col);
      chip.appendChild(nm);
      spkBar.appendChild(chip);
    });
    const add = document.createElement('button');
    add.id = 'addspk';
    add.textContent = '+ Add speaker';
    add.addEventListener('click', () => {
      newSpeaker('Speaker ' + (spkSeq + 1));
      render();
    });
    spkBar.appendChild(add);
  }

  function makeSelect(seg) {
    const sel = document.createElement('select');
    sel.className = 'spk';
    SPEAKERS.forEach((sp) => {
      const o = document.createElement('option');
      o.value = sp.key;
      o.textContent = sp.name;
      sel.appendChild(o);
    });
    const nw = document.createElement('option');
    nw.value = '__new__';
    nw.textContent = '+ New speaker…';
    sel.appendChild(nw);
    sel.value = seg.spk;
    sel.style.background = getSpk(seg.spk).color;
    sel.addEventListener('change', () => {
      if (sel.value === '__new__') {
        const name = prompt('New speaker name for this line:');
        if (name) seg.spk = newSpeaker(name).key;
        render();
        return;
      }
      seg.spk = sel.value;
      renderTranscript();
    });
    return sel;
  }

  function renderTranscript() {
    list.innerHTML = '';
    rows = [];
    SEGS.forEach((seg) => {
      const row = document.createElement('div');
      row.className = 'line';
      const ts = document.createElement('span');
      ts.className = 'ts';
      ts.textContent = fmt(seg.start);
      ts.title = 'Jump to ' + fmt(seg.start);
      ts.addEventListener('click', () => seek(seg.start));
      const tx = document.createElement('span');
      tx.className = 'tx';
      tx.textContent = seg.text;
      row.appendChild(ts);
      row.appendChild(makeSelect(seg));
      row.appendChild(tx);
      list.appendChild(row);
      rows.push(row);
    });
    activeIdx = -1;
  }

  function render() { renderSpeakerBar(); renderTranscript(); }

  function currentIndex(t) {
    let idx = -1;
    for (let i = 0; i < SEGS.length; i++) {
      if (t >= SEGS[i].start && t < SEGS[i].end) return i;
      if (t >= SEGS[i].start) idx = i;
    }
    return idx;
  }
  function sync(scroll) {
    const idx = currentIndex(player.currentTime);
    if (idx !== activeIdx) {
      if (rows[activeIdx]) rows[activeIdx].classList.remove('active');
      activeIdx = idx;
      if (rows[idx]) {
        rows[idx].classList.add('active');
        if (scroll) rows[idx].scrollIntoView({ block: 'nearest' });
      }
    }
    if (caption) {
      const seg = SEGS[idx];
      caption.innerHTML = seg ? '<span>' + seg.text.replace(/</g, '&lt;') + '</span>' : '';
    }
  }
  player.addEventListener('timeupdate', () => sync(true));
  ['seeked', 'play', 'pause', 'loadedmetadata'].forEach(
    (ev) => player.addEventListener(ev, () => sync(false))
  );

  function withSpk() { return document.getElementById('ex_spk').checked; }
  function toTxt() {
    const spk = withSpk();
    const out = [];
    let prev = null;
    SEGS.forEach((s) => {
      if (spk) {
        const n = getSpk(s.spk).name;
        if (n !== prev) { out.push('\n[' + n + ']'); prev = n; }
      }
      out.push(s.text);
    });
    return out.join('\n').trim() + '\n';
  }
  function toSrt() {
    const spk = withSpk();
    return SEGS.map((s, i) =>
      (i + 1) + '\n' + tsCode(s.start, ',') + ' --> ' + tsCode(s.end, ',') +
      '\n' + (spk ? '[' + getSpk(s.spk).name + '] ' : '') + s.text + '\n'
    ).join('\n');
  }
  function toVtt() {
    const spk = withSpk();
    const out = ['WEBVTT\n'];
    SEGS.forEach((s) => {
      out.push(
        tsCode(s.start, '.') + ' --> ' + tsCode(s.end, '.') +
        '\n' + (spk ? '<v ' + getSpk(s.spk).name + '>' : '') + s.text + '\n'
      );
    });
    return out.join('\n');
  }
  function toJson() {
    const spk = withSpk();
    const data = SEGS.map((s) => spk
      ? { start: s.start, end: s.end, speaker: getSpk(s.spk).name, text: s.text }
      : { start: s.start, end: s.end, text: s.text });
    return JSON.stringify(data, null, 2) + '\n';
  }
  function download(name, text) {
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }
  const EXPORTS = { ex_txt: toTxt, ex_srt: toSrt, ex_vtt: toVtt, ex_json: toJson };
  Object.keys(EXPORTS).forEach((id) => {
    document.getElementById(id).addEventListener('click', () => {
      download('transcript.' + id.slice(3), EXPORTS[id]());
    });
  });

  render();
"""


def _player_html(src: str, is_video: bool, segments: list[TranscriptSegment]) -> str:
    """Self-contained HTML: media player + an editable, clickable transcript.

    `src` is a `data:` URI (inlined) or an `app/static/...` URL. Editing and
    exporting are fully client-side, so tweaks never trigger a Streamlit rerun.
    """
    seg_data = [
        {
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "spk": seg.speaker,
            "text": seg.text,
        }
        for seg in segments
    ]
    order: list[str] = []
    for seg in segments:
        if seg.speaker not in order:
            order.append(seg.speaker)
    speakers = [
        {"key": spk, "name": spk, "color": SPEAKER_PALETTE[i % len(SPEAKER_PALETTE)]}
        for i, spk in enumerate(order)
    ]

    tag = "video" if is_video else "audio"
    extra = "max-height:55vh;background:#000;" if is_video else ""
    style = "width:100%;border-radius:8px;" + extra
    media_el = (
        f'<{tag} id="player" src="{src}" controls preload="metadata" '
        f'style="{style}"></{tag}>'
    )
    overlay = '<div id="caption"></div>' if is_video else ""

    data_js = (
        f"const SEGS = {json.dumps(seg_data, ensure_ascii=False)};\n"
        f"const SPEAKERS = {json.dumps(speakers, ensure_ascii=False)};\n"
        f"const PALETTE = {json.dumps(SPEAKER_PALETTE)};\n"
    )
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\"><style>"
        + _PLAYER_CSS
        + "</style></head><body>"
        + f'<div id="stage">{media_el}{overlay}</div>'
        + '<div id="speakers"></div><div id="transcript"></div>'
        + '<div id="exports">'
        + '<label title="Include speaker names in the downloaded file">'
        + '<input type="checkbox" id="ex_spk" checked> Speaker labels</label>'
        + '<button id="ex_txt">⬇ .txt</button><button id="ex_srt">⬇ .srt</button>'
        + '<button id="ex_vtt">⬇ .vtt</button><button id="ex_json">⬇ .json</button>'
        + "</div><script>"
        + data_js
        + _PLAYER_JS
        + "</script></body></html>"
    )


def render_file_browser(start_dir: str) -> None:
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
        st.error("Permission denied for this folder.")
        return

    dirs = [p for p in entries if p.is_dir() and not p.name.startswith(".")]
    files = [p for p in entries if p.is_file() and p.suffix.lower() in MEDIA_EXTS]

    PLACEHOLDER = "— select a folder or file —"
    UP = "⬆️  .. (up one level)"
    options = [PLACEHOLDER, UP]
    options += [f"📁 {p.name}" for p in dirs]
    options += [f"🎵 {p.name}" for p in files]

    choice = st.selectbox("Browse", options, key=f"browse_{cur}")

    if choice == UP:
        ss.browse_dir = str(cur.parent)
        st.rerun()
    elif choice.startswith("📁 "):
        ss.browse_dir = str(cur / choice[2:].strip())
        st.rerun()
    elif choice.startswith("🎵 "):
        ss.picked_path = str(cur / choice[2:].strip())


with st.sidebar:
    st.header("Settings")

    device, compute_type = detect_device()

    backend_options = ["Auto", "faster-whisper"]
    if mlx_available():
        backend_options.insert(1, "mlx")
    backend_label = st.selectbox(
        "Backend",
        backend_options,
        index=0,
        help="MLX uses the Apple Silicon GPU — much faster on Mac. "
        "faster-whisper uses CUDA or CPU.",
    )
    backend = "auto" if backend_label == "Auto" else backend_label
    resolved_backend = detect_backend() if backend == "auto" else backend
    diar_device = detect_diarization_device()
    st.info(
        f"Transcription: **{describe_runtime(resolved_backend, device, compute_type)}**\n\n"
        f"Diarization: **pyannote · {diar_device}**"
    )
    if not mlx_available():
        from speaker_transcribe.config import is_apple_silicon

        if is_apple_silicon():
            st.caption("💡 Install `mlx-whisper` (run `uv sync`) for faster Mac transcription.")

    hf_token = st.text_input(
        "HuggingFace token",
        value=get_hf_token() or "",
        type="password",
        help="Required for diarization. Accept the model terms on HuggingFace first.",
    )

    downloaded = downloaded_whisper_models(resolved_backend)

    def _model_label(m: str) -> str:
        mark = "✅ " if m in downloaded else "⬇️ "
        return f"{mark}{m} — {WHISPER_MODELS[m]}"

    model = st.selectbox(
        "Whisper model",
        MODEL_NAMES,
        index=MODEL_NAMES.index("small"),
        format_func=_model_label,
        help="Smaller = faster (good for testing). large-v3 = best quality. "
        "turbo/large-v3-turbo = fast & multilingual. distil-* = English only.\n\n"
        "✅ = already downloaded (ready offline) · ⬇️ = will be fetched on first use.",
    )
    if model in downloaded:
        st.caption("✅ Downloaded — ready to use right now.")
    else:
        st.caption("⬇️ Not downloaded yet — it'll be fetched on first run.")
    english_only = is_english_only_model(model)
    if english_only:
        st.caption("⚠️ This is an English-only model.")

    def _lang_label(code: str) -> str:
        return "Auto-detect" if code == "auto" else f"{WHISPER_LANGUAGES[code].title()} ({code})"

    if english_only:
        st.selectbox(
            "Language",
            ["en"],
            format_func=_lang_label,
            disabled=True,
            help="English-only model — language is fixed to English.",
        )
        language: str | None = "en"
    else:
        lang_codes = ["auto"] + sorted(WHISPER_LANGUAGES, key=lambda c: WHISPER_LANGUAGES[c])
        choice = st.selectbox(
            "Language",
            lang_codes,
            index=0,
            format_func=_lang_label,
            help="Language of the audio. Auto-detect lets Whisper guess it.",
        )
        language = None if choice == "auto" else choice

    diarize = st.checkbox(
        "Identify speakers (diarization)",
        value=bool(hf_token.strip()),
        help="Requires a HuggingFace token. Off = transcription only — every line "
        "starts as one speaker and you split them manually in the transcript editor.",
    )
    if diarize:
        col_a, col_b = st.columns(2)
        min_speakers = col_a.number_input("Min speakers", 1, 20, 2)
        max_speakers = col_b.number_input("Max speakers", 1, 20, 5)
    else:
        min_speakers, max_speakers = 1, 5
        st.caption("📝 Transcription only — no speaker labels (no token needed).")

    st.subheader("Fragment (for long files)")
    fragment_mode = st.radio(
        "Process",
        ["First N seconds", "Time range", "Whole file"],
        index=2,
    )
    start, duration = 0.0, None
    if fragment_mode == "First N seconds":
        duration = float(st.number_input("Seconds", 5, 36000, 60, step=5))
    elif fragment_mode == "Time range":
        start = float(st.number_input("Start (s)", 0, 36000, 0, step=5))
        end = float(st.number_input("End (s)", 1, 36000, 120, step=5))
        duration = max(0.0, end - start)

    beam_choice = st.selectbox(
        "Beam size",
        ["Auto", 1, 2, 3, 4, 5],
        index=0,
        help="Higher = more accurate but slower. Auto = 1 on CPU, 5 on GPU.",
    )
    beam_size = None if beam_choice == "Auto" else int(beam_choice)
    if beam_choice == "Auto":
        st.caption(f"Auto → beam size {auto_beam_size(device)} on this {device}.")
    vad_filter = st.checkbox("VAD filter (skip silence)", value=True)
    if resolved_backend == "mlx":
        st.caption("ℹ️ Beam size and VAD filter apply to faster-whisper only.")

tab_upload, tab_path = st.tabs(["📤 Upload file", "📁 Local path"])
input_path: str | None = None
cleanup_path: Path | None = None

with tab_upload:
    uploaded = st.file_uploader(
        "Audio or video file",
        type=["mp3", "wav", "m4a", "flac", "ogg", "mp4", "mov", "mkv", "webm", "avi"],
    )
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".bin"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)  # noqa: SIM115
        tmp.write(uploaded.getbuffer())
        tmp.close()
        input_path = tmp.name
        cleanup_path = Path(tmp.name)
        st.success(f"Loaded: {uploaded.name}")

with tab_path:
    if "picked_path" not in st.session_state:
        st.session_state.picked_path = ""

    default_root = "/data" if Path("/data").is_dir() else str(Path.home())
    with st.expander("📂 Browse files on this machine", expanded=False):
        render_file_browser(default_root)

    typed = st.text_input(
        "Path to an existing file on this machine",
        key="picked_path",
        help="Type a path, or pick one with the file browser above.",
    )
    if typed:
        if Path(typed).exists():
            input_path = typed
            st.success(f"Found: {typed}")
        else:
            st.error("File not found.")

run = st.button("▶️ Transcribe", type="primary", disabled=input_path is None)


def _speaker_settings(segments: list[TranscriptSegment]) -> tuple[dict[str, str], dict[str, str]]:
    """Per-speaker (alias, color) dicts, seeded with defaults in session state."""
    speakers: list[str] = []
    for seg in segments:
        if seg.speaker not in speakers:
            speakers.append(seg.speaker)

    aliases: dict[str, str] = st.session_state.setdefault("speaker_aliases", {})
    colors: dict[str, str] = st.session_state.setdefault("speaker_colors", {})
    for i, spk in enumerate(speakers):
        aliases.setdefault(spk, spk)
        colors.setdefault(spk, SPEAKER_PALETTE[i % len(SPEAKER_PALETTE)])

    with st.expander("✏️ Edit speakers", expanded=False):
        for spk in speakers:
            c1, c2 = st.columns([4, 1])
            aliases[spk] = c1.text_input(
                f"Name for {spk}", value=aliases[spk], key=f"alias_{spk}"
            ).strip() or spk
            colors[spk] = c2.color_picker("Color", value=colors[spk], key=f"color_{spk}")

    return aliases, colors


def _render_markdown_transcript(
    segments: list[TranscriptSegment], aliases: dict[str, str], colors: dict[str, str]
) -> None:
    """Static transcript fallback when no inline media preview is available."""
    prev = None
    body = []
    for seg in segments:
        if seg.speaker != prev:
            name = escape(aliases.get(seg.speaker, seg.speaker))
            color = colors.get(seg.speaker, "#000000")
            body.append(f'\n<strong style="color:{color}">{name}</strong>')
            prev = seg.speaker
        body.append(
            f'&nbsp;&nbsp;<code>{format_clock(seg.start)}</code> {escape(seg.text)}'
        )
    st.markdown("  \n".join(body), unsafe_allow_html=True)


def _player_src(info: tuple) -> str | None:
    """`data:` URI when small enough to inline, else an `app/static/...` URL."""
    _kind, _payload, name, mime, _is_video, size = info
    if size <= MAX_INLINE_BYTES:
        data = _read_media_bytes(info)
        if data is not None:
            return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    return _static_media_url(info)


def _render(segments: list[TranscriptSegment]) -> None:
    st.subheader("Transcript")

    info = _media_info()
    src = _player_src(info) if info else None
    if src:
        is_video = info[4]
        st.caption(
            "▶️ Click a timestamp to jump there. Edit names & colours, and reassign "
            "any line's speaker — handy for splitting *Unknown* into different people. "
            "Use the buttons below the transcript to download your edited result."
        )
        components.html(
            _player_html(src, is_video, segments),
            height=900 if is_video else 620,
            scrolling=True,
        )
        return

    aliases, colors = _speaker_settings(segments)
    _render_markdown_transcript(segments, aliases, colors)

    aliased = [
        TranscriptSegment(seg.start, seg.end, aliases.get(seg.speaker, seg.speaker), seg.text)
        for seg in segments
    ]

    st.subheader("Download")
    include_speakers = st.checkbox(
        "Include speaker labels", value=True, help="Off = transcript text only."
    )
    cols = st.columns(len(EXPORTERS))
    for col, fmt in zip(cols, EXPORTERS, strict=False):
        col.download_button(
            f".{fmt}",
            data=export(aliased, fmt, include_speakers),
            file_name=f"transcript.{fmt}",
            mime="text/plain",
            use_container_width=True,
        )


HF_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
if run and input_path and diarize and not hf_token.strip():
    st.error(
        "🔑 **A HuggingFace token is required for speaker diarization.**\n\n"
        "1. Create a free token at https://huggingface.co/settings/tokens\n"
        f"2. Accept the model terms at https://huggingface.co/{HF_DIARIZATION_MODEL}\n"
        "3. Paste it into **HuggingFace token** in the sidebar "
        "(or set `HF_TOKEN` in a local `.env`), then run again.\n\n"
        "Or untick **Identify speakers (diarization)** to transcribe without "
        "speaker labels — no token needed."
    )
    if cleanup_path and cleanup_path.exists():
        cleanup_path.unlink(missing_ok=True)
    run = False

if run and input_path:
    settings = Settings(
        input_path=input_path,
        start=start,
        duration=duration,
        whisper_model=model,
        language=language,
        diarize=diarize,
        min_speakers=int(min_speakers),
        max_speakers=int(max_speakers),
        backend=backend,
        beam_size=beam_size,
        vad_filter=vad_filter,
        hf_token=hf_token.strip() or None,
    )

    status = st.status("Working…", expanded=True)
    bar = st.progress(0.0, text="Preparing…")

    def _on_progress(stage: str, frac: float, eta: float | None) -> None:
        if "no live progress" in stage:
            bar.empty()
            status.write(
                "⏳ Transcribing with MLX — this runs in one shot, so there's no "
                "live %. The spinner above means it's still working."
            )
            return
        frac = min(max(float(frac), 0.0), 1.0)
        label = f"{stage}… {int(frac * 100)}%"
        if eta:
            label += f" • ~{format_eta(eta)} left"
        bar.progress(frac, text=label)

    try:
        segments = run_pipeline(
            settings,
            progress=lambda m: status.write(m),
            on_progress=_on_progress,
        )
        bar.empty()
        status.update(label="Done", state="complete", expanded=False)
        st.session_state.pop("speaker_aliases", None)
        st.session_state.pop("speaker_colors", None)
        st.session_state.segments = segments
        if uploaded is not None and cleanup_path and str(cleanup_path) == input_path:
            st.session_state.media_source = ("bytes", bytes(uploaded.getbuffer()), uploaded.name)
        else:
            st.session_state.media_source = ("path", input_path, Path(input_path).name)
        if not segments:
            st.warning("No speech detected in the selected fragment.")
    except Exception as exc:
        bar.empty()
        status.update(label="Failed", state="error")
        st.session_state.pop("segments", None)
        if isinstance(exc, RuntimeError):
            st.error(f"⚠️ {exc}")
        else:
            st.exception(exc)
    finally:
        if cleanup_path and cleanup_path.exists():
            cleanup_path.unlink(missing_ok=True)


if st.session_state.get("segments"):
    _render(st.session_state.segments)
