import fnmatch
from pathlib import Path
from types import SimpleNamespace

import pytest

from speaker_transcribe import catalog, hub
from speaker_transcribe.engines.onnx import merge_spans, token_words
from speaker_transcribe.transcript import split_by_speaker

# Weight, vocab and config files published in each GigaAM ONNX repo (int8 only).
REPO_FILES = {
    "v2": [
        "config.json",
        "v2_vocab.txt",
        "v2_ctc.int8.onnx",
        "v2_rnnt_encoder.int8.onnx",
        "v2_rnnt_decoder.int8.onnx",
        "v2_rnnt_joint.int8.onnx",
    ],
    "v3": [
        "config.json",
        "v3_vocab.txt",
        "v3_e2e_ctc_vocab.txt",
        "v3_e2e_rnnt_vocab.txt",
        "v3_ctc.int8.onnx",
        "v3_e2e_ctc.int8.onnx",
        "v3_rnnt_encoder.int8.onnx",
        "v3_rnnt_decoder.int8.onnx",
        "v3_rnnt_joint.int8.onnx",
        "v3_e2e_rnnt_encoder.int8.onnx",
        "v3_e2e_rnnt_decoder.int8.onnx",
        "v3_e2e_rnnt_joint.int8.onnx",
    ],
}


def test_token_words_groups_characters_on_spaces():
    tokens = list("да ну") + [" "]
    stamps = [0.0, 0.1, 0.2, 0.5, 0.6, 0.7]
    words = token_words(tokens, stamps, offset=10.0, end=11.0)
    assert words == [(10.0, 10.5, "да"), (10.5, 11.0, "ну")]


def test_token_words_handles_e2e_subwords_and_punctuation():
    # v3 e2e: a leading space starts a word, punctuation joins the word before it.
    tokens = [" Пр", "ивет", ",", " как", " дела", "?", " —", " Хорошо", "."]
    stamps = [0.0, 0.2, 0.4, 0.6, 0.9, 1.2, 1.4, 1.6, 2.0]
    words = token_words(tokens, stamps, offset=5.0, end=7.5)
    assert [w[2] for w in words] == ["Привет,", "как", "дела?", "—", "Хорошо."]
    assert words[0][:2] == (5.0, 5.6)
    assert words[-1][:2] == (6.6, 7.5)


def test_merge_spans_joins_short_pauses_up_to_the_model_limit():
    rate = 10  # samples per second, to keep the numbers readable
    spans = [(0, 20), (23, 40), (60, 80), (82, 230), (232, 290)]
    # 0.3 s pause → joined; 2 s pause → kept apart; joining past 20 s → kept apart.
    assert merge_spans(spans, rate=rate) == [(0, 40), (60, 230), (232, 290)]
    assert merge_spans([], rate=rate) == []


def test_token_words_empty():
    assert token_words([], [], 0.0, 1.0) == []


def test_split_by_speaker_cuts_at_speaker_change():
    diar = [
        {"start": 0.0, "end": 2.0, "speaker": "S0"},
        {"start": 2.0, "end": 4.0, "speaker": "S1"},
    ]
    sp_map = {"S0": "Speaker 1", "S1": "Speaker 2"}
    words = [(0.0, 1.0, "привет"), (1.0, 2.0, "как"), (2.1, 3.0, "нормально"), (3.0, 4.0, "да")]
    segs = list(split_by_speaker(0.0, 4.0, "привет как нормально да", words, diar, sp_map))
    assert [(s.speaker, s.text) for s in segs] == [
        ("Speaker 1", "привет как"),
        ("Speaker 2", "нормально да"),
    ]
    assert segs[1].start == 2.1


def test_split_by_speaker_keeps_unmatched_words_with_main_speaker():
    diar = [{"start": 0.0, "end": 3.0, "speaker": "S0"}]
    sp_map = {"S0": "Speaker 1"}
    words = [(0.0, 1.0, "раз"), (5.0, 6.0, "два")]  # second word outside every turn
    segs = list(split_by_speaker(0.0, 6.0, "раз два", words, diar, sp_map))
    assert [(s.speaker, s.text) for s in segs] == [("Speaker 1", "раз два")]


def test_split_by_speaker_without_diarization_keeps_piece_whole():
    segs = list(split_by_speaker(1.0, 2.0, "текст", [(1.0, 2.0, "текст")], [], {}))
    assert [(s.start, s.end, s.text) for s in segs] == [(1.0, 2.0, "текст")]


def _fake_cache(monkeypatch, repo_id: str, files: list[str]) -> None:
    snap = Path("/cache/snapshots/abc")
    rev = SimpleNamespace(
        snapshot_path=snap,
        files=[SimpleNamespace(file_path=snap / name) for name in files],
    )
    info = SimpleNamespace(repos=[SimpleNamespace(repo_id=repo_id, revisions=[rev])])
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda: info)


def test_repo_files_cached_matches_patterns(monkeypatch):
    repo = catalog.GIGAAM_REPOS["v2"]
    _fake_cache(
        monkeypatch,
        repo,
        ["config.json", "v2_vocab.txt", "v2_rnnt_encoder.int8.onnx", "v2_rnnt_decoder.int8.onnx"],
    )
    assert hub.repo_files_cached(repo, catalog.GIGAAM_FILES["gigaam-v2-rnnt"])
    assert not hub.repo_files_cached(repo, catalog.GIGAAM_FILES["gigaam-v2-ctc"])
    assert not hub.repo_files_cached("other/repo", ["config.json"])


def test_v3_variants_are_cached_independently(monkeypatch):
    repo = catalog.GIGAAM_REPOS["v3"]
    _fake_cache(
        monkeypatch,
        repo,
        ["config.json", "v3_e2e_rnnt_vocab.txt"]
        + [f"v3_e2e_rnnt_{part}.int8.onnx" for part in ("encoder", "decoder", "joint")],
    )
    assert hub.repo_files_cached(repo, catalog.GIGAAM_FILES["gigaam-v3-e2e-rnnt"])
    # The plain RNN-T needs its own weights, not the e2e ones.
    assert not hub.repo_files_cached(repo, catalog.GIGAAM_FILES["gigaam-v3-rnnt"])
    assert not hub.repo_files_cached(repo, catalog.GIGAAM_FILES["gigaam-v3-e2e-ctc"])


def test_gigaam_engine_is_russian_only():
    engine = catalog.ENGINES["gigaam"]
    assert engine.languages == {"ru": "Russian"}
    assert set(engine.models) == set(catalog.GIGAAM_FILES)
    assert next(iter(engine.models)) == "gigaam-v3-e2e-rnnt"  # the default
    for model in engine.models:
        assert catalog.gigaam_version(model) in catalog.GIGAAM_VERSIONS
        assert catalog.gigaam_repo_for(model) == catalog.GIGAAM_REPOS[catalog.gigaam_version(model)]


@pytest.mark.parametrize("model", list(catalog.GIGAAM_MODELS))
def test_downloaded_files_are_exactly_what_onnx_asr_loads(model):
    """Each onnx-asr file pattern matches one downloaded file, so loading works offline."""
    gigaam = pytest.importorskip("onnx_asr.models.gigaam")
    model_class = {
        "gigaam-v2-ctc": gigaam.GigaamV2Ctc,
        "gigaam-v2-rnnt": gigaam.GigaamV2Rnnt,
        "gigaam-v3-ctc": gigaam.GigaamV2Ctc,
        "gigaam-v3-rnnt": gigaam.GigaamV2Rnnt,
        "gigaam-v3-e2e-ctc": gigaam.GigaamV3E2eCtc,
        "gigaam-v3-e2e-rnnt": gigaam.GigaamV3E2eRnnt,
    }[model]
    published = REPO_FILES[catalog.gigaam_version(model)]
    downloaded = [
        f for f in published if any(fnmatch.fnmatch(f, p) for p in catalog.GIGAAM_FILES[model])
    ]
    for pattern in model_class._get_model_files(catalog.ONNX_QUANTIZATION).values():
        # onnx-asr globs inside the snapshot; with every variant cached it must
        # still find exactly one file per pattern.
        assert len(fnmatch.filter(published, pattern)) == 1, pattern
        assert len(fnmatch.filter(downloaded, pattern)) == 1, pattern


def test_pyannote_telemetry_off_by_default():
    import os

    assert os.environ["PYANNOTE_METRICS_ENABLED"] == "false"


def test_audio_too_short_for_the_vad_has_no_speech():
    np = pytest.importorskip("numpy")
    pytest.importorskip("onnx_asr")
    from speaker_transcribe.engines.onnx import speech_pieces

    assert speech_pieces(np.zeros(256, dtype=np.float32)) == []
