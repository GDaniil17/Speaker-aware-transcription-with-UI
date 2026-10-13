"""The UI must prepare every snapshot needed before transcription starts."""

import threading

import pytest

from speaker_transcribe import model_downloads as downloads
from speaker_transcribe.catalog import (
    GIGAAM_FILES,
    GIGAAM_REPOS,
    GIGAAM_VAD_REPO,
    QWEN3_ALIGNER_REPO,
    QWEN3_ASR_REPO,
)


def test_qwen_needs_asr_and_aligner(monkeypatch):
    cached = {QWEN3_ASR_REPO}
    monkeypatch.setattr(downloads, "repo_in_cache", lambda repo: repo in cached)

    assert not downloads.model_is_cached("qwen3-asr", "Qwen3-ASR-1.7B")
    cached.add(QWEN3_ALIGNER_REPO)
    assert downloads.model_is_cached("qwen3-asr", "Qwen3-ASR-1.7B")


def test_gigaam_fetches_only_selected_weights_and_vad():
    repos = downloads.model_repos("gigaam", "gigaam-v2-ctc")
    assert repos == [
        (GIGAAM_REPOS["v2"], GIGAAM_FILES["gigaam-v2-ctc"]),
        (GIGAAM_VAD_REPO, downloads.GIGAAM_VAD_FILES),
    ]


def test_gigaam_v3_downloads_from_the_v3_repo():
    repos = downloads.model_repos("gigaam", "gigaam-v3-e2e-rnnt")
    assert repos == [
        (GIGAAM_REPOS["v3"], GIGAAM_FILES["gigaam-v3-e2e-rnnt"]),
        (GIGAAM_VAD_REPO, downloads.GIGAAM_VAD_FILES),
    ]


def test_download_progress_spans_multiple_repos(monkeypatch):
    calls = []
    progress = []

    def fake_download(repo, callback, patterns):
        calls.append((repo, patterns))
        callback("Downloading model", 0.5, 10.0)

    monkeypatch.setattr(downloads, "ensure_repo_downloaded", fake_download)
    downloads.download_model(
        "qwen3-asr", "Qwen3-ASR-1.7B", on_progress=lambda *args: progress.append(args)
    )

    assert calls == [(QWEN3_ASR_REPO, None), (QWEN3_ALIGNER_REPO, None)]
    assert [item[1] for item in progress] == [0.25, 0.5, 0.75, 1.0]


def test_refresh_reattaches_instead_of_starting_a_parallel_download(monkeypatch):
    downloads._TASKS.clear()
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def fake_download(engine, model, backend, on_progress):
        calls.append((engine, model, backend))
        on_progress("repo · Downloading model", 0.25, None)
        entered.set()
        assert release.wait(2)

    monkeypatch.setattr(downloads, "download_model", fake_download)
    monkeypatch.setattr(downloads, "model_is_cached", lambda *_: True)
    first = downloads.start_model_download("gigaam", "gigaam-v2-ctc", "auto")
    assert entered.wait(2)

    assert downloads.start_model_download("gigaam", "gigaam-v2-ctc", "auto") is first
    assert downloads.get_model_download_task("gigaam", "gigaam-v2-ctc", "auto") is first
    assert downloads.active_model_download_task() is first
    with pytest.raises(RuntimeError, match="Another model download"):
        downloads.start_model_download("gigaam", "gigaam-v2-rnnt", "auto")

    release.set()
    assert first.done.wait(2)
    assert calls == [("gigaam", "gigaam-v2-ctc", "auto")]
    assert downloads.active_model_download_task() is None


def test_failed_download_can_be_retried(monkeypatch):
    downloads._TASKS.clear()
    attempts = []

    def fake_download(*args):
        attempts.append(args)
        if len(attempts) == 1:
            raise RuntimeError("network interrupted")

    monkeypatch.setattr(downloads, "download_model", fake_download)
    monkeypatch.setattr(downloads, "model_is_cached", lambda *_: True)
    first = downloads.start_model_download("gigaam", "gigaam-v2-ctc", "auto")
    assert first.done.wait(2)
    assert first.snapshot()[3] == "network interrupted"

    second = downloads.start_model_download("gigaam", "gigaam-v2-ctc", "auto")
    assert second is not first
    assert second.done.wait(2)
    assert second.snapshot()[3] is None
    assert len(attempts) == 2
