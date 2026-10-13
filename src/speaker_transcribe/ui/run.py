"""Running the pipeline from the UI with a live step list and progress bar."""

from __future__ import annotations

import threading
import time

import streamlit as st

from speaker_transcribe.catalog import DIARIZATION_MODEL
from speaker_transcribe.config import Settings
from speaker_transcribe.pipeline import run_pipeline
from speaker_transcribe.transcript import TranscriptSegment
from speaker_transcribe.ui.formatting import (
    format_duration,
    format_step_duration,
    format_total_duration,
)
from speaker_transcribe.ui.i18n import progress_text, tr


class _LiveStatus:
    """Collects pipeline callbacks from a worker thread and repaints the UI.

    The pipeline blocks for minutes between its per-chunk callbacks, so it runs
    in a worker thread while the main thread repaints on a timer. That keeps the
    elapsed clock ticking instead of only updating when the percentage changes.
    """

    def __init__(self, status, bar_slot) -> None:
        self.status = status
        self.bar_slot = bar_slot
        self.bar = bar_slot.progress(0.0, text=tr("Preparing…"))
        # Written by the worker thread, read by the main thread.
        self._lock = threading.Lock()
        self._logs: list[tuple[str, float]] = []
        self._stage: str | None = None
        self._frac = 0.0
        self._eta: float | None = None
        self._oneshot: str | None = None
        # Main-thread state.
        self._written = 0
        self._oneshot_shown = False
        self._stage_key: str | None = None
        self._stage_started = 0.0
        self._step_slot = None
        self._step_text = ""
        self._step_raw = ""
        self._step_started = 0.0
        self.steps: list[tuple[str, float]] = []

    # --- worker thread ---------------------------------------------------------

    def log(self, msg: str) -> None:
        with self._lock:
            self._logs.append((msg, time.monotonic()))

    def progress(self, stage: str, frac: float, eta: float | None) -> None:
        with self._lock:
            if "no live progress" in stage:
                self._oneshot = stage
                return
            self._oneshot = None
            self._stage = stage
            self._frac = min(max(float(frac), 0.0), 1.0)
            self._eta = eta

    # --- main thread -----------------------------------------------------------

    def finish_step(self, at: float) -> None:
        if self._step_slot is None:
            return
        took = at - self._step_started
        self.steps.append((self._step_raw, took))
        self._step_slot.markdown(f"✓ {self._step_text} · **{format_step_duration(took)}**")
        self._step_slot = None

    def refresh(self) -> None:
        with self._lock:
            stage, frac, eta, oneshot = self._stage, self._frac, self._eta, self._oneshot
            new_logs = self._logs[self._written :]
            self._written = len(self._logs)

        for msg, at in new_logs:
            self.finish_step(at)
            if msg.startswith("Done —"):
                continue
            self._step_text = progress_text(msg)
            self._step_raw = msg
            self._step_started = at
            self._step_slot = self.status.empty()
        if self._step_slot is not None:
            running = format_step_duration(time.monotonic() - self._step_started)
            self._step_slot.markdown(f"⏳ {self._step_text} · {running}")

        if oneshot is not None:
            if not self._oneshot_shown:
                self.bar_slot.empty()
                self.bar = None
                self.status.caption(tr("This step reports time but no live percentage."))
                self._oneshot_shown = True
            return
        if stage is None:
            return

        now = time.monotonic()
        # Start the clock when a new stage begins; keep it running across the
        # whole stage (frac sits at 0 for the entire first chunk). Key on the
        # text before any " (…)" suffix so a live byte readout that changes
        # every tick (e.g. "Downloading model (1.4 GB / 2.4 GB)") doesn't keep
        # resetting the elapsed clock.
        stage_key = stage.split(" (", 1)[0]
        if self._stage_key != stage_key:
            self._stage_key, self._stage_started = stage_key, now
        elapsed = now - self._stage_started
        if eta is not None:
            remaining: float | None = float(eta)
        elif frac > 0.02:
            remaining = elapsed * (1.0 - frac) / frac
        else:
            remaining = None
        label = (
            f"{progress_text(stage)}… {int(frac * 100)}% · "
            f"{format_duration(elapsed)} {tr('elapsed')}"
        )
        if remaining is not None:
            label += f" / ~{format_duration(remaining)} {tr('left')}"
        if self.bar is None:
            # A one-shot stage (e.g. the whole-file attempt on MLX) took the bar
            # away; chunked retries do report progress, so bring it back.
            self.bar = self.bar_slot.progress(0.0, text=label)
            self._oneshot_shown = False
        self.bar.progress(frac, text=label)


def show_token_needed() -> None:
    st.error(
        f"🔑 {tr('A Hugging Face token is needed once to download the speaker model.')}\n\n"
        f"[Token](https://huggingface.co/settings/tokens) · "
        f"[Model terms](https://huggingface.co/{DIARIZATION_MODEL})\n\n"
        f"{tr('Paste the token in Settings, or turn off speaker identification.')}"
    )


def show_last_run() -> None:
    """The step timings of the previous run, collapsed."""
    timeline = st.session_state.get("run_timeline")
    if not timeline:
        return
    with st.status(
        f"{tr('Last run')} · {format_duration(timeline['total'])}",
        state="complete",
        expanded=False,
    ):
        for message, seconds in timeline["steps"]:
            st.markdown(f"✓ {progress_text(message)} · **{format_step_duration(seconds)}**")
        st.markdown(f"**{tr('Total time')} · {format_total_duration(timeline['total'])}**")


def run_with_progress(settings: Settings) -> list[TranscriptSegment] | None:
    """Run the pipeline, showing live progress. Returns None when it failed."""
    run_started = time.monotonic()
    status = st.status(tr("Working…"), expanded=True)
    live = _LiveStatus(status, st.empty())
    box: dict[str, object] = {}

    def worker() -> None:
        try:
            box["segments"] = run_pipeline(settings, progress=live.log, on_progress=live.progress)
        except Exception as exc:  # re-raised on the main thread below
            box["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        while thread.is_alive():
            live.refresh()
            time.sleep(0.5)
        thread.join()
        live.refresh()  # flush any final logs / progress
        live.finish_step(time.monotonic())
        if "error" in box:
            raise box["error"]  # type: ignore[misc]
    except Exception as exc:
        live.bar_slot.empty()
        status.update(
            label=f"{tr('Failed')} · {format_duration(time.monotonic() - run_started)}",
            state="error",
        )
        st.session_state.pop("run_timeline", None)
        if isinstance(exc, RuntimeError):
            st.error(f"⚠️ {exc}")
        else:
            st.exception(exc)
        return None

    segments: list[TranscriptSegment] = box.get("segments") or []  # type: ignore[assignment]
    live.bar_slot.empty()
    total_seconds = time.monotonic() - run_started
    st.session_state.run_timeline = {"total": total_seconds, "steps": live.steps}
    status.markdown(f"**{tr('Total time')} · {format_total_duration(total_seconds)}**")
    status.update(
        label=f"{tr('Done')} · {format_duration(total_seconds)}",
        state="complete",
        expanded=False,
    )
    return segments
