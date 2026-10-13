"""Download sizes shown before a model is fetched."""

from types import SimpleNamespace

import pytest

from speaker_transcribe import hub
from speaker_transcribe import model_downloads as downloads
from speaker_transcribe.catalog import GIGAAM_REPOS, GIGAAM_VAD_REPO
from speaker_transcribe.ui import formatting


@pytest.fixture(autouse=True)
def fresh_size_cache():
    hub._SIZES.clear()
    yield
    hub._SIZES.clear()


def fake_hub(monkeypatch, files: dict[str, int] | None, calls: list | None = None):
    """Make HfApi().model_info return `files` (name -> bytes), or fail when None."""

    def model_info(self, repo_id, files_metadata=False, timeout=None):
        if calls is not None:
            calls.append(repo_id)
        if files is None:
            raise OSError("offline")
        siblings = [SimpleNamespace(rfilename=n, size=s) for n, s in files.items()]
        return SimpleNamespace(siblings=siblings)

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", model_info)


def test_remote_size_counts_only_matching_files_and_is_cached(monkeypatch):
    calls = []
    fake_hub(monkeypatch, {"config.json": 1, "v3_ctc.int8.onnx": 200, "v3_ctc.onnx": 800}, calls)

    assert hub.remote_size("org/repo", ["config.json", "*.int8.onnx"]) == 201
    assert hub.remote_size("org/repo", ["config.json", "*.int8.onnx"]) == 201
    assert hub.remote_size("org/repo") == 1001
    assert calls == ["org/repo", "org/repo"]  # one lookup per pattern set


def test_failed_lookup_is_retried_only_after_a_while(monkeypatch):
    calls = []
    fake_hub(monkeypatch, None, calls)
    clock = [1000.0]
    monkeypatch.setattr(hub.time, "monotonic", lambda: clock[0])

    assert hub.remote_size("org/repo") is None
    assert hub.remote_size("org/repo") is None
    assert len(calls) == 1  # an offline UI doesn't wait on every rerun

    clock[0] += hub._SIZE_RETRY_SECONDS + 1
    fake_hub(monkeypatch, {"model.bin": 5}, calls)
    assert hub.remote_size("org/repo") == 5


def test_download_size_skips_snapshots_already_cached(monkeypatch):
    sizes = {GIGAAM_REPOS["v3"]: 230_000_000, GIGAAM_VAD_REPO: 2_000_000}
    monkeypatch.setattr(downloads, "remote_size", lambda repo, patterns=None: sizes[repo])
    cached: set[str] = set()
    monkeypatch.setattr(downloads, "repo_files_cached", lambda repo, patterns: repo in cached)

    assert downloads.download_size("gigaam", "gigaam-v3-e2e-rnnt") == 232_000_000
    cached.add(GIGAAM_VAD_REPO)  # e.g. shared with GigaAM v2, downloaded earlier
    assert downloads.download_size("gigaam", "gigaam-v3-e2e-rnnt") == 230_000_000
    cached.add(GIGAAM_REPOS["v3"])
    assert downloads.download_size("gigaam", "gigaam-v3-e2e-rnnt") == 0


def test_download_size_is_unknown_when_any_lookup_fails(monkeypatch):
    monkeypatch.setattr(downloads, "repo_files_cached", lambda repo, patterns: False)
    monkeypatch.setattr(
        downloads,
        "remote_size",
        lambda repo, patterns=None: None if repo == GIGAAM_VAD_REPO else 1,
    )
    assert downloads.download_size("gigaam", "gigaam-v2-ctc") is None


@pytest.mark.parametrize(
    ("num_bytes", "english", "russian"),
    [
        (1_610_000_000, "1.6 GB", "1,6 ГБ"),
        (232_000_000, "232 MB", "232 МБ"),
        (31_000, "1 MB", "1 МБ"),
    ],
)
def test_format_size(monkeypatch, num_bytes, english, russian):
    monkeypatch.setattr(formatting, "is_russian", lambda: False)
    assert formatting.format_size(num_bytes) == english
    monkeypatch.setattr(formatting, "is_russian", lambda: True)
    assert formatting.format_size(num_bytes) == russian
