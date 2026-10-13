from speaker_transcribe.export import export, to_json, to_srt, to_txt, to_vtt
from speaker_transcribe.transcript import TranscriptSegment

SEGMENTS = [
    TranscriptSegment(0.0, 2.5, "Speaker 1", "Hello there."),
    TranscriptSegment(2.5, 4.0, "Speaker 1", "How are you?"),
    TranscriptSegment(4.0, 6.0, "Speaker 2", "I am fine."),
]


def test_txt_groups_consecutive_speakers():
    out = to_txt(SEGMENTS)
    assert out.count("[Speaker 1]") == 1
    assert "[Speaker 2]" in out
    assert "Hello there." in out


def test_json_roundtrip():
    import json

    data = json.loads(to_json(SEGMENTS))
    assert len(data) == 3
    assert data[0]["speaker"] == "Speaker 1"
    assert data[2]["text"] == "I am fine."


def test_srt_format():
    out = to_srt(SEGMENTS)
    assert "1\n00:00:00,000 --> 00:00:02,500" in out
    assert "[Speaker 1] Hello there." in out


def test_vtt_format():
    out = to_vtt(SEGMENTS)
    assert out.startswith("WEBVTT")
    assert "00:00:04.000 --> 00:00:06.000" in out
    assert "<v Speaker 2>I am fine." in out


def test_export_dispatch_and_error():
    assert export(SEGMENTS, "txt") == to_txt(SEGMENTS)
    try:
        export(SEGMENTS, "xml")
    except ValueError as exc:
        assert "Unknown format" in str(exc)
    else:
        raise AssertionError("expected ValueError")
