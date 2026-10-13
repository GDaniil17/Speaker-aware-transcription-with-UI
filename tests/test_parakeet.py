"""Parakeet works out of the box on ONNX and prefers a faster backend when installed."""

import fnmatch

import pytest

from speaker_transcribe import catalog, engine_install, runtime
from speaker_transcribe import model_downloads as downloads


def installed(monkeypatch, modules: set[str], apple_silicon: bool = True):
    monkeypatch.setattr(runtime, "is_apple_silicon", lambda: apple_silicon)
    monkeypatch.setattr(
        runtime.importlib.util,
        "find_spec",
        lambda name: object() if name in modules else None,
    )


@pytest.mark.parametrize(
    ("modules", "apple_silicon", "expected"),
    [
        ({"onnx_asr"}, True, "onnx"),
        ({"onnx_asr", "parakeet_mlx"}, True, "mlx"),
        ({"onnx_asr", "parakeet_mlx"}, False, "onnx"),  # parakeet-mlx needs Apple Silicon
        ({"onnx_asr", "nemo"}, False, "nemo"),
        (set(), False, None),
    ],
)
def test_backend_preference(monkeypatch, modules, apple_silicon, expected):
    installed(monkeypatch, modules, apple_silicon)
    assert runtime.parakeet_backend() == expected
    assert runtime.engine_available("parakeet") == (expected is not None)


def test_onnx_backend_downloads_the_onnx_model_and_vad(monkeypatch):
    monkeypatch.setattr(downloads, "parakeet_backend", lambda: "onnx")
    assert downloads.model_repos("parakeet", catalog.PARAKEET_MODEL) == [
        (catalog.PARAKEET_REPOS["onnx"], catalog.PARAKEET_ONNX_FILES),
        (catalog.VAD_REPO, catalog.VAD_FILES),
    ]


@pytest.mark.parametrize("backend", ["mlx", "nemo"])
def test_optional_backends_download_their_own_repo(monkeypatch, backend):
    monkeypatch.setattr(downloads, "parakeet_backend", lambda: backend)
    assert downloads.model_repos("parakeet", catalog.PARAKEET_MODEL) == [
        (catalog.PARAKEET_REPOS[backend], None)
    ]


def test_parakeet_needs_no_install():
    assert catalog.ENGINES["parakeet"].install_extra == ""
    assert engine_install.engine_install_hint("parakeet") == "uv sync"


def test_onnx_files_are_exactly_what_onnx_asr_loads():
    nemo = pytest.importorskip("onnx_asr.models.nemo")
    published = [  # int8 files of istupakov/parakeet-tdt-0.6b-v3-onnx, plus fp32 ones
        "config.json",
        "vocab.txt",
        "nemo128.onnx",
        "encoder-model.onnx",
        "encoder-model.onnx.data",
        "encoder-model.int8.onnx",
        "decoder_joint-model.onnx",
        "decoder_joint-model.int8.onnx",
    ]
    downloaded = [
        f for f in published if any(fnmatch.fnmatch(f, p) for p in catalog.PARAKEET_ONNX_FILES)
    ]
    patterns = nemo.NemoConformerTdt._get_model_files(catalog.ONNX_QUANTIZATION).values()
    for pattern in patterns:
        assert len(fnmatch.filter(downloaded, pattern)) == 1, pattern
    assert "encoder-model.onnx.data" not in downloaded  # the 2.4 GB fp32 weights
