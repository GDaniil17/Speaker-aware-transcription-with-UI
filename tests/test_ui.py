"""The Streamlit page renders in both interface languages and in basic mode."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from speaker_transcribe.transcript import TranscriptSegment
from speaker_transcribe.ui.player import player_html

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _run(monkeypatch, basic: bool = False) -> AppTest:
    if basic:
        monkeypatch.setenv("SPEAKER_TRANSCRIBE_BASIC", "1")
    else:
        monkeypatch.delenv("SPEAKER_TRANSCRIBE_BASIC", raising=False)
    app = AppTest.from_file(APP, default_timeout=60)
    app.run()
    assert not app.exception, app.exception
    return app


def test_page_renders_in_english(monkeypatch):
    app = _run(monkeypatch)
    assert "Speaker-aware transcription" in app.title[0].value
    transcribe = next(b for b in app.button if "Transcribe" in b.label)
    assert transcribe.disabled  # nothing to transcribe yet


def test_switching_to_russian_translates_the_page(monkeypatch):
    app = _run(monkeypatch)
    app.sidebar.selectbox(key="ui_language").set_value("RU").run()
    assert not app.exception, app.exception
    assert "Расшифровка с разделением говорящих" in app.title[0].value


def test_basic_mode_offers_whisper_only(monkeypatch):
    app = _run(monkeypatch, basic=True)
    engine = app.sidebar.selectbox(key="engine_select")
    assert engine.options == ["Whisper"]
    assert app.sidebar.toggle(key="diarize").disabled


@pytest.mark.parametrize("text", ["</script><b>x</b>", "a < b && c > d"])
def test_transcript_text_cannot_break_out_of_the_player_script(text):
    html = player_html(None, False, [TranscriptSegment(0.0, 1.0, "Speaker 1", text)])
    script = html.split("<script>", 1)[1]
    assert script.count("</script>") == 1  # only the real closing tag


def test_gigaam_offers_a_version_choice(monkeypatch):
    app = _run(monkeypatch)
    app.sidebar.selectbox(key="engine_select").set_value("gigaam").run()
    assert not app.exception, app.exception
    version = app.sidebar.radio(key="gigaam_version")
    assert version.value == "v3"
    v3_models = app.sidebar.selectbox(key="model_gigaam_v3")
    assert v3_models.value == "gigaam-v3-e2e-rnnt"
    assert all("-v3-" in m for m in v3_models.options)

    version.set_value("v2").run()
    assert not app.exception, app.exception
    v2_models = app.sidebar.selectbox(key="model_gigaam_v2")
    assert v2_models.options == ["gigaam-v2-rnnt", "gigaam-v2-ctc"]
