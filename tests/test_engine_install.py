"""Engine installs must stay inside the project environment and use fixed targets."""

import pytest

from speaker_transcribe import engine_install


def test_install_command_is_a_fixed_uv_sync_in_the_project_venv(monkeypatch, tmp_path):
    monkeypatch.setattr(engine_install.sys, "prefix", str(tmp_path / ".venv"))
    monkeypatch.setattr(engine_install.shutil, "which", lambda name: f"/usr/bin/{name}")

    assert engine_install.install_command("qwen3-asr", tmp_path) == [
        "/usr/bin/uv",
        "sync",
        "--inexact",
        "--extra",
        "qwen3-asr",
    ]
    assert engine_install.install_command("whisper", tmp_path) is None
    assert engine_install.install_command("unknown", tmp_path) is None


def test_install_hint_works_without_make():
    assert (
        engine_install.engine_install_hint("sensevoice") == "uv sync --inexact --extra sensevoice"
    )
    assert engine_install.engine_install_hint("gigaam") == "uv sync"


def test_install_command_rejects_other_python_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(engine_install.sys, "prefix", str(tmp_path / "system-python"))
    monkeypatch.setattr(engine_install.shutil, "which", lambda name: f"/usr/bin/{name}")

    assert engine_install.install_command("qwen3-asr", tmp_path) is None


def test_install_command_needs_uv(monkeypatch, tmp_path):
    monkeypatch.setattr(engine_install.sys, "prefix", str(tmp_path / ".venv"))
    monkeypatch.setattr(engine_install.shutil, "which", lambda name: None)

    assert engine_install.install_command("sensevoice", tmp_path) is None


def test_install_engine_streams_output_and_reports_failure(monkeypatch):
    calls = []

    class FakeProcess:
        stdout = iter(["Resolved 10 packages\n", "Could not resolve dependencies\n"])

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def wait(self):
            return 1

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr(
        engine_install, "install_command", lambda engine: ["uv", "sync", "--extra", "qwen3-asr"]
    )
    monkeypatch.setattr(engine_install.subprocess, "Popen", fake_popen)
    output = []

    with pytest.raises(RuntimeError, match="Could not resolve dependencies"):
        engine_install.install_engine("qwen3-asr", output.append)

    assert output == ["Resolved 10 packages", "Could not resolve dependencies"]
    assert calls[0][0] == ["uv", "sync", "--extra", "qwen3-asr"]
    assert calls[0][1]["cwd"] == engine_install.PROJECT_ROOT
    assert calls[0][1]["env"]["UV_NO_PROGRESS"] == "1"
