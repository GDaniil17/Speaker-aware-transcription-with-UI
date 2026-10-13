"""Transcript editor: media player plus clickable, editable transcript lines."""

from __future__ import annotations

import html
import json
from functools import cache
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from speaker_transcribe.transcript import TranscriptSegment
from speaker_transcribe.ui.i18n import tr
from speaker_transcribe.ui.media import current_media, mp3_fallback_url, player_src

ASSETS = Path(__file__).resolve().parent / "assets"

SPEAKER_PALETTE = [
    "#4363d8",
    "#3cb44b",
    "#e6194B",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#9A6324",
]


@cache
def _asset(name: str) -> str:
    return (ASSETS / name).read_text(encoding="utf-8")


def _labels() -> dict[str, str]:
    """Interface text used inside the component, in the current language."""
    return {
        "speakers": tr("Speakers:"),
        "addSpeaker": tr("Add speaker"),
        "speaker": tr("Speaker"),
        "newSpeaker": tr("New speaker…"),
        "newSpeakerPrompt": tr("New speaker name for this line:"),
        "jumpTo": tr("Jump to"),
        "fallbackPreparing": tr(
            "This browser can't play this video format. Preparing an MP3 preview of the audio…"
        ),
        "fallbackPlaying": tr(
            "This browser can't play this video format, so an MP3 of its audio plays instead."
        ),
    }


def player_html(
    src: str | None,
    is_video: bool,
    segments: list[TranscriptSegment],
    fallback: str | None = None,
) -> str:
    """Self-contained HTML: media player + an editable, clickable transcript.

    `src` is a `data:` URI (inlined) or an `app/static/...` URL; `fallback` is an
    MP3 URL the player switches to if the browser can't decode the video.
    Editing and exporting are fully client-side, so tweaks never trigger a rerun.
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
    order = list(dict.fromkeys(seg.speaker for seg in segments))
    speakers = [
        {"key": spk, "name": spk, "color": SPEAKER_PALETTE[i % len(SPEAKER_PALETTE)]}
        for i, spk in enumerate(order)
    ]

    stage = ""
    if src:
        tag = "video" if is_video else "audio"
        extra = "max-height:55vh;background:#000;" if is_video else ""
        alt = f' data-fallback="{fallback}"' if fallback else ""
        media_el = (
            f'<{tag} id="player" src="{src}" controls preload="metadata" '
            f'style="width:100%;border-radius:8px;{extra}"{alt}></{tag}>'
        )
        overlay = '<div id="caption"></div>' if is_video else ""
        stage = f'<div id="stage">{media_el}{overlay}</div>'

    # "</" would end the inline <script> early if a transcript line contained it.
    def as_js(value: object) -> str:
        return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")

    data_js = (
        f"const SEGS = {as_js(seg_data)};\n"
        f"const SPEAKERS = {as_js(speakers)};\n"
        f"const PALETTE = {as_js(SPEAKER_PALETTE)};\n"
        f"const LABELS = {as_js(_labels())};\n"
    )
    include_title = html.escape(tr("Include speaker names in the downloaded file"))
    return (
        '<!doctype html><html><head><meta charset="utf-8"><style>'
        + _asset("player.css")
        + "</style></head><body>"
        + stage
        + '<div id="speakers"></div><div id="transcript"></div>'
        + '<div id="exports">'
        + f'<label title="{include_title}">'
        + f'<input type="checkbox" id="ex_spk" checked> {html.escape(tr("Speaker labels"))}</label>'
        + '<button id="ex_txt">⬇ .txt</button><button id="ex_srt">⬇ .srt</button>'
        + '<button id="ex_vtt">⬇ .vtt</button><button id="ex_json">⬇ .json</button>'
        + "</div><script>"
        + data_js
        + _asset("player.js")
        + "</script></body></html>"
    )


def render_transcript(segments: list[TranscriptSegment]) -> None:
    """The transcript section: player (when the media can be shown) and editor."""
    st.subheader(tr("Transcript"))

    media = current_media()
    src = player_src(media) if media else None
    if media and src:
        st.caption(tr("Click a timestamp to play from there. Edit speakers and lines below."))
        fallback = mp3_fallback_url(media) if media.is_video else None
        components.html(
            player_html(src, media.is_video, segments, fallback),
            height=900 if media.is_video else 620,
            scrolling=True,
        )
        return

    st.caption(tr("Edit speakers and lines below, then download your transcript."))
    components.html(player_html(None, False, segments), height=620, scrolling=True)
