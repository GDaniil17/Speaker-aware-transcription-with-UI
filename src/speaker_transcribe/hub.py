"""Hugging Face cache checks and model downloads with live byte progress."""

from __future__ import annotations

import fnmatch
import importlib
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

from speaker_transcribe.catalog import DIARIZATION_MODEL, whisper_repo_for

ProgressBarFn = Callable[[str, float, float | None], None]


# Download sizes looked up on the Hub: {(repo, patterns): (bytes or None, when)}.
# A size is kept for the life of the process; a failed lookup (offline, Hub
# down) is retried after _SIZE_RETRY_SECONDS, so an offline UI isn't slowed down
# by a request on every rerun.
_SIZES: dict[tuple[str, tuple[str, ...] | None], tuple[int | None, float]] = {}
_SIZES_LOCK = threading.Lock()
_SIZE_RETRY_SECONDS = 300.0


def remote_size(repo_id: str, patterns: list[str] | None = None) -> int | None:
    """Bytes a snapshot download of `repo_id` (only `patterns`, if given) fetches.

    None when the Hub cannot be reached. Gated repos report sizes without a token.
    """
    key = (repo_id, tuple(patterns) if patterns else None)
    with _SIZES_LOCK:
        cached = _SIZES.get(key)
    if cached is not None:
        size, when = cached
        if size is not None or time.monotonic() - when < _SIZE_RETRY_SECONDS:
            return size
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id, files_metadata=True, timeout=10)
        size: int | None = sum(
            sibling.size or 0
            for sibling in info.siblings or []
            if not patterns or any(fnmatch.fnmatch(sibling.rfilename, p) for p in patterns)
        )
    except Exception:
        size = None
    with _SIZES_LOCK:
        _SIZES[key] = (size, time.monotonic())
    return size


def repo_in_cache(repo_id: str) -> bool:
    """True when `repo_id` is already present in the local HuggingFace cache."""
    try:
        from huggingface_hub import scan_cache_dir

        return repo_id in {repo.repo_id for repo in scan_cache_dir().repos}
    except Exception:
        return False


def repo_files_cached(repo_id: str, patterns: list[str]) -> bool:
    """True when every pattern matches a file of `repo_id` in the local HF cache."""
    try:
        from huggingface_hub import scan_cache_dir

        repo = next(r for r in scan_cache_dir().repos if r.repo_id == repo_id)
    except Exception:
        return False
    names = {
        f.file_path.relative_to(rev.snapshot_path).as_posix()
        for rev in repo.revisions
        for f in rev.files
    }
    return all(any(fnmatch.fnmatch(name, pat) for name in names) for pat in patterns)


def local_repo_snapshot(repo_id: str, required: str = "config.yaml") -> Path | None:
    """Local snapshot folder of `repo_id` holding `required`, or None if not downloaded.

    Never touches the network, so it works without a token for gated repos.
    """
    try:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(repo_id, local_files_only=True))
    except Exception:
        return None
    return path if (path / required).is_file() else None


def diarization_model_cached(repo_id: str = DIARIZATION_MODEL) -> bool:
    """True when the pyannote pipeline is downloaded, so no HF token is needed."""
    return local_repo_snapshot(repo_id) is not None


def ensure_whisper_model_downloaded(
    model: str,
    backend: str,
    on_progress: ProgressBarFn | None = None,
) -> None:
    """Download a Whisper model snapshot when it is missing from the local HF cache."""
    repo_id = whisper_repo_for(model, backend)
    if repo_id:
        ensure_repo_downloaded(repo_id, on_progress)


def _human_bytes(num: float) -> str:
    """Format a byte count as a compact human-readable string (e.g. '1.4 GB')."""
    # Decimal units, as the Hub lists file sizes (and the UI shows them before a download).
    for unit in ("B", "KB", "MB"):
        if num < 1000:
            return f"{num:.0f} {unit}"
        num /= 1000
    return f"{num:.1f} GB"


def _download_stage(cur: float, tot: float) -> str:
    """Stage label with live byte progress so a slow download reads as moving,
    not frozen, even while the rounded percentage holds steady."""
    if tot <= 0:
        return "Downloading model"
    return f"Downloading model ({_human_bytes(cur)} / {_human_bytes(tot)})"


def ensure_repo_downloaded(
    repo_id: str,
    on_progress: ProgressBarFn | None = None,
    allow_patterns: list[str] | None = None,
) -> None:
    """Download a HuggingFace repo snapshot when missing, reporting download progress.

    With `allow_patterns`, only the matching files are fetched (and checked for).
    """
    from huggingface_hub import snapshot_download

    if allow_patterns:
        if repo_files_cached(repo_id, allow_patterns):
            return
    elif repo_in_cache(repo_id):
        return
    if not on_progress:
        snapshot_download(repo_id, allow_patterns=allow_patterns)
        return

    # huggingface_hub creates the real per-file byte bars from
    # `huggingface_hub.utils.tqdm.tqdm` — NOT from snapshot_download's
    # `tqdm_class` argument (which only wraps the outer "Fetching N files"
    # counter). Patch that class so we observe actual byte progress.
    # NB: `import huggingface_hub.utils.tqdm as m` binds the `tqdm` *class*, not
    # the submodule, because `huggingface_hub.utils.__init__` does
    # `from .tqdm import tqdm`, rebinding that attribute on the package. Import
    # the module explicitly so we can patch its module-level `tqdm`.
    hf_tqdm_mod = importlib.import_module("huggingface_hub.utils.tqdm")

    base_tqdm = hf_tqdm_mod.tqdm

    state = {"frac": 0.0, "eta": None, "cur": 0.0, "tot": 0.0}
    state_lock = threading.Lock()
    done = threading.Event()
    errors: list[BaseException] = []
    # Aggregate progress across the per-file byte bars.
    bars: dict[int, tuple[float, float]] = {}
    # Sliding window of (monotonic_time, bytes_downloaded) for a *recent*
    # throughput estimate. A whole-run average over-/under-shoots badly on a
    # slow tail, making the ETA jump around and the bar look frozen.
    samples: deque[tuple[float, float]] = deque(maxlen=64)
    on_progress("Downloading model", 0.0, None)

    def _recompute() -> None:
        cur = sum(c for c, _ in bars.values())
        tot = sum(t for _, t in bars.values())
        frac = min(cur / tot, 1.0) if tot else 0.0
        now = time.monotonic()
        samples.append((now, cur))
        # Speed over the sampling window (≥1s span) → ETA on remaining bytes.
        eta: float | None = None
        if len(samples) >= 2:
            t0, b0 = samples[0]
            dt, db = now - t0, cur - b0
            if dt >= 1.0 and db > 0:
                eta = (tot - cur) / (db / dt)
        state.update(frac=frac, eta=eta, cur=cur, tot=tot)

    class DownloadProgress(base_tqdm):
        # Suppress terminal output without disabling tracking: `disable=True`
        # makes tqdm.update() early-return, freezing self.n at 0.
        def refresh(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return None

        def display(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return None

        def update(self, n=1):  # noqa: ANN001
            displayed = super().update(n)
            # Only byte-unit bars are real file downloads; ignore the "Fetching
            # N files" counter so the fraction reflects downloaded bytes.
            if getattr(self, "unit", "") == "B" and self.total:
                with state_lock:
                    bars[id(self)] = (float(self.n or 0.0), float(self.total))
                    _recompute()
            return displayed

    def download() -> None:
        hf_tqdm_mod.tqdm = DownloadProgress
        try:
            snapshot_download(repo_id, tqdm_class=DownloadProgress, allow_patterns=allow_patterns)
        except BaseException as exc:
            errors.append(exc)
        finally:
            hf_tqdm_mod.tqdm = base_tqdm
            done.set()

    thread = threading.Thread(target=download, name="hf-model-download", daemon=True)
    thread.start()

    while not done.wait(0.2):
        with state_lock:
            frac, eta, cur, tot = (
                float(state["frac"]),
                state["eta"],
                float(state["cur"]),
                float(state["tot"]),
            )
        on_progress(_download_stage(cur, tot), frac, eta)

    thread.join()
    if errors:
        raise errors[0]

    on_progress("Downloading model", 1.0, 0.0)
