"""Install a selected engine into the project's existing virtual environment."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

from speaker_transcribe.catalog import ENGINES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_INSTALL_LOCK = threading.Lock()


def engine_install_extra(engine: str) -> str:
    """The project extra needed for an optional engine, if any."""
    return ENGINES[engine].install_extra


def engine_install_hint(engine: str) -> str:
    """The terminal command that installs `engine`'s optional dependencies."""
    extra = engine_install_extra(engine)
    if extra:
        return f"uv sync --inexact --extra {extra}"
    return "uv sync" if engine in ("gigaam", "parakeet") else ""  # onnx-asr is core


def install_command(engine: str, project_root: Path = PROJECT_ROOT) -> list[str] | None:
    """Return a project-scoped install command, or None outside its uv venv."""
    if engine not in ENGINES or engine == "whisper":
        return None
    if Path(sys.prefix).resolve() != (project_root / ".venv").resolve():
        return None
    uv = shutil.which("uv")
    if not uv:
        return None
    command = [uv, "sync", "--inexact"]
    extra = engine_install_extra(engine)
    if extra:
        command += ["--extra", extra]
    return command


def install_engine(engine: str, on_line: Callable[[str], None]) -> None:
    """Run the allowlisted installer and stream its output to the UI."""
    command = install_command(engine)
    if command is None:
        raise RuntimeError("Run the app from the project's .venv with uv available.")
    if not _INSTALL_LOCK.acquire(blocking=False):
        raise RuntimeError("Another engine installation is already running.")
    try:
        env = os.environ.copy()
        env.update(UV_NO_PROGRESS="1", NO_COLOR="1")
        recent: list[str] = []
        with subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                message = line.strip()
                if message:
                    recent.append(message)
                    recent = recent[-8:]
                    on_line(message)
            code = process.wait()
        if code:
            details = recent[-1] if recent else f"exit code {code}"
            raise RuntimeError(details)
    finally:
        _INSTALL_LOCK.release()
