"""The parent side of the transcription worker: heartbeats, kills, relaying."""

import importlib.util
from pathlib import Path

import pytest

from speaker_transcribe.chunking import StalledError
from speaker_transcribe.config import Settings
from speaker_transcribe.transcript import TranscriptSegment
from speaker_transcribe.worker import _ChildRunner, transcription_runner


class _FakeConn:
    """Stands in for the pipe to the child: canned messages, then silence."""

    def __init__(self, messages: list[tuple] | None = None) -> None:
        self.messages = list(messages or [])
        self.sent: list[object] = []
        self.closed = False
        self.eof = False

    def send(self, payload: object) -> None:
        self.sent.append(payload)

    def poll(self, timeout: float | None = None) -> bool:  # noqa: ARG002
        return bool(self.messages) or self.eof

    def recv(self) -> tuple:
        if self.messages:
            return self.messages.pop(0)
        raise EOFError

    def close(self) -> None:
        self.closed = True


class _FakeProc:
    """Stands in for the child process, without starting one."""

    def __init__(self, alive: bool = True) -> None:
        self.alive = alive
        self.killed = False
        self.exitcode = None if alive else -9

    def is_alive(self) -> bool:
        return self.alive

    def kill(self) -> None:
        self.killed, self.alive = True, False

    def join(self, timeout: float | None = None) -> None:
        return None


def _runner_with(messages: list[tuple], stall_timeout: float) -> tuple[_ChildRunner, _FakeProc]:
    runner = _ChildRunner(stall_timeout)
    proc = _FakeProc()
    runner._proc = proc
    runner._conn = _FakeConn(messages)
    return runner, proc


def _run(runner: _ChildRunner, **kwargs):
    return runner.run(Path("input.wav"), Settings(input_path="input.wav"), [], {}, **kwargs)


def test_parent_relays_messages_and_returns_rows():
    runner, proc = _runner_with(
        [
            ("log", "Transcribing…"),
            ("progress", "Transcribing", 0.5, 12.0),
            ("segment", 0.0, 1.5, "Speaker 1", "hello"),
            ("done",),
        ],
        stall_timeout=30.0,
    )

    logs: list[str] = []
    stages: list[tuple[str, float, float | None]] = []
    rows = _run(runner, log=logs.append, on_progress=lambda *args: stages.append(args))

    assert rows == [(0.0, 1.5, "Speaker 1", "hello")]
    assert logs == ["Transcribing…"]
    assert stages == [("Transcribing", 0.5, 12.0)]
    assert not proc.killed


def test_parent_gives_up_on_a_silent_worker():
    runner, proc = _runner_with([], stall_timeout=0.5)

    with pytest.raises(StalledError, match="no progress"):
        _run(runner)

    assert proc.killed  # so the next attempt starts from a clean process


def test_progress_keeps_a_slow_worker_alive():
    # Six progress reports spread over more than the 1s timeout: each one is a
    # heartbeat, so the run is slow rather than stalled.
    runner, proc = _runner_with(
        [("progress", "Transcribing", 0.2, None)] * 6
        + [("segment", 0.0, 1.0, "Speaker 1", "hi"), ("done",)],
        stall_timeout=1.0,
    )
    rows = _run(runner)

    assert rows == [(0.0, 1.0, "Speaker 1", "hi")]
    assert not proc.killed


def test_parent_reports_a_worker_that_died():
    runner, _ = _runner_with([], stall_timeout=60.0)
    runner._proc = _FakeProc(alive=False)

    with pytest.raises(RuntimeError, match="out of memory"):
        _run(runner)


def test_a_closed_pipe_is_reported_as_a_dead_worker():
    runner, _ = _runner_with([], stall_timeout=60.0)
    runner._conn.eof = True  # poll() says "ready", recv() raises EOFError

    with pytest.raises(RuntimeError, match="died"):
        _run(runner)


def test_engine_errors_come_back_as_runtime_errors():
    runner, proc = _runner_with(
        [("error", "RuntimeError: model not found", "Traceback…")], stall_timeout=60.0
    )

    with pytest.raises(RuntimeError, match="model not found"):
        _run(runner)

    assert proc.killed


def test_no_stall_timeout_means_no_child_process(monkeypatch):
    calls: list[Path] = []

    def fake_transcribe(wav_path, settings, diar, sp_map, progress, on_progress=None):
        calls.append(wav_path)
        yield TranscriptSegment(0.0, 1.0, "Speaker 1", "hello")

    monkeypatch.setattr("speaker_transcribe.engines.transcribe", fake_transcribe)

    with transcription_runner(0.0) as run_job:
        rows = run_job(Path("input.wav"), Settings(input_path="input.wav"), [], {}, None, None)

    assert rows == [(0.0, 1.0, "Speaker 1", "hello")]
    assert calls == [Path("input.wav")]


@pytest.mark.skipif(
    importlib.util.find_spec("qwen_asr") is not None,
    reason="needs an engine that is not installed, so the child fails fast",
)
def test_a_real_child_process_reports_its_error_back():
    """End-to-end over the process boundary: spawn, run, relay the failure."""
    settings = Settings(input_path="input.wav", engine="qwen3-asr")

    with transcription_runner(60.0) as run_job, pytest.raises(RuntimeError, match="qwen-asr"):
        run_job(Path("input.wav"), settings, [], {}, None, None)
