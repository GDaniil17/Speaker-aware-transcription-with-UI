"""SenseVoice words come back whole; punctuation and contractions are rejoined."""

from speaker_transcribe import catalog
from speaker_transcribe import model_downloads as downloads
from speaker_transcribe.engines.sensevoice import timed_words
from speaker_transcribe.transcript import split_by_speaker


def test_contractions_and_punctuation_join_the_word_before():
    words = ["Today", "we", "'", "re", "talking", ",", "right", "?"]
    stamps = [
        [0, 300],
        [300, 400],
        [400, 420],
        [420, 500],
        [500, 900],
        [900, 920],
        [1000, 1300],
        [1300, 1320],
    ]
    assert timed_words(words, stamps, offset=10.0) == [
        (10.0, 10.3, "Today"),
        (10.3, 10.5, "we're"),
        (10.5, 10.92, "talking,"),
        (11.0, 11.32, "right?"),
    ]


def test_cjk_characters_stay_separate_and_rejoin_without_spaces():
    words = ["大", "家", "好", "。", "你", "好"]
    stamps = [[0, 100], [100, 200], [200, 300], [300, 320], [2000, 2100], [2100, 2200]]
    timed = timed_words(words, stamps, offset=0.0)
    assert [w[2] for w in timed] == ["大", "家", "好。", "你", "好"]
    diar = [
        {"start": 0.0, "end": 1.0, "speaker": "A"},
        {"start": 1.5, "end": 3.0, "speaker": "B"},
    ]
    segs = list(split_by_speaker(0.0, 2.2, "大家好。你好", timed, diar, {"A": "S1", "B": "S2"}))
    assert [(s.speaker, s.text) for s in segs] == [("S1", "大家好。"), ("S2", "你好")]


def test_one_speaker_keeps_the_models_text():
    timed = [(0.0, 0.5, "Thanks"), (0.5, 1.0, "for"), (1.0, 1.5, "having"), (1.5, 2.0, "me,")]
    diar = [{"start": 0.0, "end": 3.0, "speaker": "A"}]
    segs = list(
        split_by_speaker(0.0, 2.0, "Thanks for having me, the idea…", timed, diar, {"A": "S1"})
    )
    assert [(s.start, s.end, s.speaker, s.text) for s in segs] == [
        (0.0, 2.0, "S1", "Thanks for having me, the idea…")
    ]


def test_sensevoice_downloads_the_model_and_the_shared_vad():
    assert downloads.model_repos("sensevoice", catalog.SENSEVOICE_MODEL) == [
        (catalog.SENSEVOICE_REPO, None),
        (catalog.VAD_REPO, catalog.VAD_FILES),
    ]
