"""Running one transcription call so it can be given up on.

Transcription happens in a child process rather than in this one. That is what
makes stall recovery possible: a Python thread stuck inside a model's decode
loop cannot be interrupted, but a child process can be killed, the audio cut
smaller and the work restarted (see `chunking.run_with_fallback`).

The child stays alive between calls, so a model is loaded once per run and
reloaded only after a stall forced a kill. With `stall_timeout <= 0` there is
nothing to enforce, so calls run in this process exactly as they used to.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from speaker_transcribe.chunking import StalledError, format_clock
from speaker_transcribe.config import Settings

# One transcribed segment as it crosses the process boundary: start, end,
# speaker, text. Kept as a plain tuple so this module stays free of model types
# (and out of an import cycle with `engines`).
Row = tuple[float, float, str, str]

LogFn = Callable[[str], None]
ProgressFn = Callable[[str, float, float | None], None]

# How often the parent wakes up to re-check the stall clock and the child.
_POLL_SECONDS = 0.25

# Starting a worker briefly edits the `__main__` module (see `_hidden_main`),
# so two runs may not do it at once.
_START_LOCK = threading.Lock()


def _process_context() -> mp.context.BaseContext:
    """The start method for workers: `forkserver` where it is safe, else `spawn`.

    Neither forks this process, which matters because the parent has already
    imported torch and may hold a diarization model. `forkserver` additionally
    forks its children from a small clean server process, so restarting a worker
    after a stall stays cheap.

    macOS always gets `spawn`: a child forked without exec cannot use XPC, and
    Metal compiles its kernels over XPC, so MLX aborts there with "Unable to
    load kernel … XPC_ERROR_CONNECTION_INVALID".
    """
    if sys.platform != "darwin" and "forkserver" in mp.get_all_start_methods():
        ctx = mp.get_context("forkserver")
        # Preload our own module rather than the default `__main__`: children
        # need it anyway, and it keeps their start-up cheap.
        ctx.set_forkserver_preload(["speaker_transcribe.engines"])
        return ctx
    return mp.get_context("spawn")


@contextmanager
def _hidden_main() -> Iterator[None]:
    """Keep a new worker from re-importing the caller's `__main__` module.

    multiprocessing hands every child the path of the main module and has it
    imported there, which re-runs a plain script that has no
    `if __name__ == "__main__":` guard — the console scripts and the Streamlit
    app are fine, a bare script or a notebook is not. Our worker only ever needs
    `speaker_transcribe`, so the main module is hidden while the child is
    created. On Windows that attribute is needed for multiprocessing's own
    `.exe` check, so it stays and an unguarded script falls back to in-process.
    """
    main = sys.modules.get("__main__")
    if main is None or sys.platform == "win32":
        yield
        return

    spec = getattr(main, "__spec__", None)
    file = getattr(main, "__file__", None)
    main.__spec__ = None
    if file is not None:
        del main.__file__
    try:
        yield
    finally:
        main.__spec__ = spec
        if file is not None:
            main.__file__ = file


def _rows_in_process(
    wav_path: Path,
    settings: Settings,
    diar: list[dict],
    sp_map: dict,
    log: LogFn | None = None,
    on_progress: ProgressFn | None = None,
) -> list[Row]:
    """Transcribe here and now — no isolation, no stall detection."""
    from speaker_transcribe.engines import transcribe

    return [
        (seg.start, seg.end, seg.speaker, seg.text)
        for seg in transcribe(wav_path, settings, diar, sp_map, log, on_progress)
    ]


def _child_main(conn, parent_pid: int) -> None:
    """Child process: transcribe each job that arrives, stream messages back.

    Every message doubles as a heartbeat, which is how the parent tells a slow
    run from a stalled one.
    """
    from speaker_transcribe.engines import transcribe

    while True:
        if not conn.poll(5.0):
            if os.getppid() != parent_pid:  # parent is gone — don't linger
                return
            continue
        try:
            job = conn.recv()
        except EOFError:
            return
        if job is None:
            return

        def log(msg: str) -> None:
            conn.send(("log", msg))

        def progress(stage: str, frac: float, eta: float | None) -> None:
            conn.send(("progress", stage, float(frac), eta))

        try:
            for seg in transcribe(
                Path(job["wav"]), job["settings"], job["diar"], job["sp_map"], log, progress
            ):
                conn.send(("segment", seg.start, seg.end, seg.speaker, seg.text))
            conn.send(("done",))
        except BaseException as exc:  # noqa: BLE001 — reported to the parent instead
            conn.send(("error", f"{type(exc).__name__}: {exc}", traceback.format_exc()))


class _ChildRunner:
    """Runs transcription jobs in a killable child process, one at a time."""

    def __init__(self, stall_timeout: float) -> None:
        self._stall_timeout = stall_timeout
        self._ctx = _process_context()
        self._proc: mp.process.BaseProcess | None = None
        self._conn = None
        self._in_process = False  # set when a child cannot be started at all

    def _start(self) -> None:
        self._conn, child_conn = self._ctx.Pipe(duplex=True)
        self._proc = self._ctx.Process(
            target=_child_main,
            args=(child_conn, os.getpid()),
            name="transcription-worker",
            # Not a daemon: some engines start their own worker processes, which
            # a daemonic parent is not allowed to do. `close()` cleans up, and
            # the child exits by itself if this process disappears.
            daemon=False,
        )
        with _START_LOCK, _hidden_main():
            self._proc.start()
        # Only the child needs its end; dropping it here makes the pipe report
        # end-of-file as soon as the child dies.
        child_conn.close()

    def _drop_connection(self) -> None:
        conn, self._conn = self._conn, None
        try:
            if conn is not None:
                conn.close()
        except Exception:  # noqa: BLE001 — cleanup is best-effort
            pass

    def _kill(self) -> None:
        """Drop the child; the next job starts a fresh one."""
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.kill()
                proc.join(timeout=10.0)
            except Exception:  # noqa: BLE001 — cleanup is best-effort
                pass
        self._drop_connection()

    def _died(self) -> RuntimeError:
        proc = self._proc
        code = proc.exitcode if proc is not None else None
        self._kill()
        return RuntimeError(
            f"the transcription worker process died (exit code {code}) — "
            "it may have run out of memory"
        )

    def close(self) -> None:
        proc, conn = self._proc, self._conn
        if proc is None:
            self._drop_connection()
            return
        try:
            if conn is not None:
                conn.send(None)  # ask for a clean exit
            proc.join(timeout=5.0)
        except Exception:  # noqa: BLE001
            pass
        if proc.is_alive():
            self._kill()
        else:
            self._proc = None
            self._drop_connection()

    def run(
        self,
        wav_path: Path,
        settings: Settings,
        diar: list[dict],
        sp_map: dict,
        log: LogFn | None = None,
        on_progress: ProgressFn | None = None,
    ) -> list[Row]:
        if not self._in_process and self._proc is None:
            try:
                self._start()
            except Exception as exc:  # noqa: BLE001 — degrade instead of failing the run
                if log:
                    log(
                        f"⚠️ Could not start the transcription worker process ({exc}); "
                        "running in this process — stalls can't be recovered from."
                    )
                self._in_process = True
        if self._in_process:
            return _rows_in_process(wav_path, settings, diar, sp_map, log, on_progress)

        conn = self._conn
        if conn is None:  # pragma: no cover - _start sets it
            raise RuntimeError("transcription worker is not running")
        conn.send({"wav": str(wav_path), "settings": settings, "diar": diar, "sp_map": sp_map})

        rows: list[Row] = []
        last_beat = time.monotonic()
        while True:
            if not conn.poll(_POLL_SECONDS):
                quiet = time.monotonic() - last_beat
                if self._stall_timeout > 0 and quiet >= self._stall_timeout:
                    self._kill()
                    raise StalledError(
                        f"no progress for {format_clock(quiet)} "
                        f"(stall timeout {format_clock(self._stall_timeout)})"
                    ) from None
                proc = self._proc
                if proc is not None and not proc.is_alive():
                    raise self._died() from None
                continue

            try:
                message = conn.recv()
            except EOFError:
                raise self._died() from None

            last_beat = time.monotonic()
            kind = message[0]
            if kind == "segment":
                rows.append((message[1], message[2], message[3], message[4]))
            elif kind == "progress":
                if on_progress:
                    on_progress(message[1], message[2], message[3])
            elif kind == "log":
                if log:
                    log(message[1])
            elif kind == "done":
                return rows
            elif kind == "error":
                # The child survives its own exceptions, but its state may not
                # be worth reusing (a half-loaded model, exhausted memory), so
                # the next job starts in a fresh process.
                self._kill()
                raise RuntimeError(message[1])


RunUnitFn = Callable[..., list[Row]]


@contextmanager
def transcription_runner(stall_timeout: float) -> Iterator[RunUnitFn]:
    """Yield `run(wav, settings, diar, sp_map, log, on_progress) -> rows`.

    Isolated in a killable child process when `stall_timeout` is positive, and
    plain in-process calls otherwise.
    """
    if stall_timeout <= 0:
        yield _rows_in_process
        return

    runner = _ChildRunner(stall_timeout)
    try:
        yield runner.run
    finally:
        runner.close()
