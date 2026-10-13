"""Cache checks and explicit downloads for selectable ASR models."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from speaker_transcribe.catalog import (
    GIGAAM_FILES,
    GIGAAM_VAD_FILES,
    GIGAAM_VAD_REPO,
    PARAKEET_REPOS,
    QWEN3_ALIGNER_REPO,
    QWEN3_ASR_REPO,
    SENSEVOICE_REPO,
    SENSEVOICE_VAD_REPO,
    gigaam_repo_for,
    mlx_repo_for,
    whisper_repo_for,
)
from speaker_transcribe.hub import (
    ensure_repo_downloaded,
    remote_size,
    repo_files_cached,
    repo_in_cache,
)
from speaker_transcribe.runtime import is_apple_silicon

DownloadProgress = Callable[[str, float, float | None], None]


@dataclass
class ModelDownloadTask:
    """Progress shared by all Streamlit sessions for one download."""

    engine: str
    model: str
    backend: str
    started_at: float = field(default_factory=time.monotonic)
    stage: str = "Preparing…"
    fraction: float = 0.0
    eta: float | None = None
    error: str | None = None
    done: threading.Event = field(default_factory=threading.Event, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def report(self, stage: str, fraction: float, eta: float | None) -> None:
        with self.lock:
            self.stage = stage
            self.fraction = min(max(fraction, 0.0), 1.0)
            self.eta = eta

    def snapshot(self) -> tuple[str, float, float | None, str | None, bool, float]:
        with self.lock:
            return (
                self.stage,
                self.fraction,
                self.eta,
                self.error,
                self.done.is_set(),
                time.monotonic() - self.started_at,
            )


_TASKS: dict[tuple[str, str, str], ModelDownloadTask] = {}
_TASKS_LOCK = threading.Lock()


def get_model_download_task(engine: str, model: str, backend: str) -> ModelDownloadTask | None:
    """Return a current or recently completed task for this exact selection."""
    with _TASKS_LOCK:
        return _TASKS.get((engine, model, backend))


def active_model_download_task() -> ModelDownloadTask | None:
    """Only one UI download may run at a time (the HF progress hook is global)."""
    with _TASKS_LOCK:
        return next((task for task in _TASKS.values() if not task.done.is_set()), None)


def start_model_download(engine: str, model: str, backend: str) -> ModelDownloadTask:
    """Start once in a server-owned thread so browser refreshes can reattach."""
    key = (engine, model, backend)
    with _TASKS_LOCK:
        active = next((task for task in _TASKS.values() if not task.done.is_set()), None)
        if active is not None:
            if (active.engine, active.model, active.backend) == key:
                return active
            raise RuntimeError("Another model download is already running.")
        task = ModelDownloadTask(engine, model, backend)
        _TASKS[key] = task

    def work() -> None:
        try:
            download_model(engine, model, backend, task.report)
            if not model_is_cached(engine, model, backend):
                raise RuntimeError("Downloaded model files are not complete in the cache.")
        except Exception as exc:
            with task.lock:
                task.error = str(exc)
        finally:
            task.done.set()

    threading.Thread(target=work, name=f"model-download-{engine}", daemon=True).start()
    return task


def model_repos(
    engine: str, model: str, backend: str = "auto"
) -> list[tuple[str, list[str] | None]]:
    """All Hugging Face snapshots needed by the selected ASR model."""
    if engine == "whisper":
        repo = mlx_repo_for(model) if backend == "mlx" else whisper_repo_for(model, backend)
        return [(repo, None)] if repo else []
    if engine == "parakeet":
        platform = "mlx" if is_apple_silicon() else "nemo"
        return [(PARAKEET_REPOS[platform], None)]
    if engine == "qwen3-asr":
        return [(QWEN3_ASR_REPO, None), (QWEN3_ALIGNER_REPO, None)]
    if engine == "sensevoice":
        return [(SENSEVOICE_REPO, None), (SENSEVOICE_VAD_REPO, None)]
    if engine == "gigaam":
        return [
            (gigaam_repo_for(model), GIGAAM_FILES[model]),
            (GIGAAM_VAD_REPO, GIGAAM_VAD_FILES),
        ]
    raise ValueError(f"Unknown engine: {engine}")


def model_is_cached(engine: str, model: str, backend: str = "auto") -> bool:
    """Check the selected model and its known auxiliary snapshots without network I/O."""
    repos = model_repos(engine, model, backend)
    return bool(repos) and all(
        repo_files_cached(repo, patterns) if patterns else repo_in_cache(repo)
        for repo, patterns in repos
    )


def download_size(engine: str, model: str, backend: str = "auto") -> int | None:
    """Bytes still to download for the selected model, or None if the Hub is unreachable.

    Snapshots already in the local cache (e.g. a VAD model another engine shares)
    don't count.
    """
    total = 0
    for repo, patterns in model_repos(engine, model, backend):
        if repo_files_cached(repo, patterns) if patterns else repo_in_cache(repo):
            continue
        size = remote_size(repo, patterns)
        if size is None:
            return None
        total += size
    return total


def download_model(
    engine: str, model: str, backend: str = "auto", on_progress: DownloadProgress | None = None
) -> None:
    """Fetch missing snapshots, mapping each repo onto one overall progress bar."""
    repos = model_repos(engine, model, backend)
    if not repos:
        raise RuntimeError(f"No download source is known for {engine} / {model}.")
    count = len(repos)
    for index, (repo, patterns) in enumerate(repos):

        def report(
            stage: str,
            fraction: float,
            eta: float | None,
            repo_id: str = repo,
            position: int = index,
        ) -> None:
            if on_progress:
                on_progress(f"{repo_id} · {stage}", (position + fraction) / count, eta)

        ensure_repo_downloaded(repo, report, patterns)
        if on_progress:
            on_progress(f"{repo} · ready", (index + 1) / count, 0.0)
