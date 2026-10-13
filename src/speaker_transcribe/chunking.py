"""Stall recovery: retry a stalled transcription in progressively smaller chunks.

A long run can stop making progress — the engine loops on a repetitive stretch
of audio, or one huge decode never returns. The way out is to stop, cut the
audio into chunks and transcribe them one at a time: a chunk that stalls again
is cut smaller still, and every chunk that does succeed stays in the transcript.

`run_with_fallback` walks that ladder of chunk sizes. It holds no audio and
loads no models — the caller passes a `run_unit` callback that transcribes one
span of audio and raises `StalledError` when that span makes no progress — so
the fallback logic is testable without an ASR engine.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")

# Chunk sizes (seconds) tried in order once a run stalls: 5 min, then 2, then 1.
DEFAULT_CHUNK_LADDER: tuple[float, ...] = (300.0, 120.0, 60.0)

# Seconds of no progress before a run counts as stalled.
DEFAULT_STALL_TIMEOUT: float = 600.0

# Anything shorter than this is too small to transcribe usefully, so smaller
# rungs are dropped from the ladder.
MIN_CHUNK_SECONDS: float = 5.0

# A trailing sliver shorter than this is folded into the chunk before it.
_MIN_TAIL_SECONDS: float = 1.0


class StalledError(RuntimeError):
    """A span of audio made no progress within the stall timeout."""


def format_clock(seconds: float) -> str:
    """Seconds as `m:ss`, or `h:mm:ss` past the hour."""
    seconds = max(int(seconds), 0)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_chunk_size(seconds: float) -> str:
    """A chunk size as `5 min` / `1.5 min` / `30 s`."""
    if seconds >= 60:
        return f"{seconds / 60:g} min"
    return f"{seconds:g} s"


def parse_ladder(text: str) -> tuple[float, ...]:
    """Parse a chunk ladder written in minutes (`"5, 2, 1"`) into seconds.

    Commas, semicolons and whitespace all separate values; decimals use a dot
    (`"0.5"` = 30 s). An empty string means "no fallback". Raises ValueError on
    anything that is not a positive number.
    """
    ladder: list[float] = []
    for part in (p for p in re.split(r"[,;\s]+", (text or "").strip()) if p):
        try:
            minutes = float(part)
        except ValueError:
            raise ValueError(f"'{part}' is not a number of minutes.") from None
        if minutes <= 0:
            raise ValueError("Chunk sizes must be larger than 0 minutes.")
        ladder.append(round(minutes * 60.0, 3))
    return tuple(ladder)


def format_ladder(ladder: Sequence[float]) -> str:
    """Render a ladder of seconds back as minutes (`"5, 2, 1"`) for the UI/CLI."""
    return ", ".join(f"{size / 60:g}" for size in ladder)


def split_region(start: float, end: float, size: float) -> list[tuple[float, float]]:
    """Split `[start, end)` into consecutive spans of `size` seconds.

    The last span holds the remainder; a remainder under a second is folded into
    the span before it rather than becoming a chunk of its own.
    """
    if end <= start:
        return []
    if size <= 0 or end - start <= size:
        return [(start, end)]

    spans: list[tuple[float, float]] = []
    cursor = start
    while cursor < end:
        stop = min(cursor + size, end)
        if end - stop < _MIN_TAIL_SECONDS:
            stop = end
        spans.append((cursor, stop))
        cursor = stop
    return spans


def merge_regions(regions: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping or adjacent spans, so a retry covers each region once."""
    merged: list[list[float]] = []
    for start, end in sorted(regions):
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def describe_regions(regions: Sequence[tuple[float, float]]) -> str:
    """Spans as `0:00–5:00, 12:30–15:00`, for log lines."""
    return ", ".join(f"{format_clock(start)}–{format_clock(end)}" for start, end in regions)


@dataclass(frozen=True)
class Unit:
    """One span of audio handed to the engine as a single transcription call."""

    start: float
    end: float
    index: int  # 1-based position among the units of this ladder level
    count: int  # how many units this level has
    chunk_size: float | None  # None when the span is transcribed unsplit

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.0)

    @property
    def short_label(self) -> str:
        """Compact label for the progress bar."""
        if self.chunk_size is None:
            return "whole file"
        return f"chunk {self.index}/{self.count}"

    @property
    def label(self) -> str:
        """Label with timecodes, for log lines."""
        span = f"{format_clock(self.start)}–{format_clock(self.end)}"
        if self.chunk_size is None:
            return f"whole file ({span})"
        return f"chunk {self.index}/{self.count} ({span})"


def _units_for(regions: Sequence[tuple[float, float]], size: float | None) -> list[Unit]:
    """The units one ladder level covers `regions` with."""
    spans: list[tuple[float, float]] = []
    for start, end in regions:
        spans.extend(split_region(start, end, size) if size else [(start, end)])
    return [
        Unit(start=start, end=end, index=i + 1, count=len(spans), chunk_size=size)
        for i, (start, end) in enumerate(spans)
    ]


def levels_for(ladder: Sequence[float], whole_file_first: bool = True) -> list[float | None]:
    """Chunk sizes to try in order; `None` means "the whole span in one call"."""
    levels: list[float | None] = [None] if whole_file_first else []
    levels += [float(size) for size in ladder if size and size >= MIN_CHUNK_SECONDS]
    return levels or [None]


def run_with_fallback(
    total: float,
    ladder: Sequence[float],
    run_unit: Callable[[Unit], list[T]],
    log: Callable[[str], None] | None = None,
    whole_file_first: bool = True,
    skip_failed: bool = True,
) -> list[T]:
    """Transcribe `total` seconds of audio, cutting it smaller whenever it stalls.

    The first attempt runs the whole span in one call (unless `whole_file_first`
    is off). Each span that stalls or errors is retried at the next chunk size
    in `ladder`; spans that succeeded are kept and never redone. Results from a
    stalled span are dropped — a stall usually means the engine was looping, so
    the smaller chunks re-transcribe that audio from scratch.

    When the ladder runs out, remaining spans are skipped with a warning, or
    raise `StalledError` when `skip_failed` is off. If nothing was transcribed
    at all, the failure that caused it is raised instead.
    """

    def say(msg: str) -> None:
        if log:
            log(msg)

    levels = levels_for(ladder, whole_file_first)
    done: list[T] = []
    pending: list[tuple[float, float]] = [(0.0, float(total))]
    first_error: BaseException | None = None

    for level, size in enumerate(levels):
        if size is not None:
            say(f"Transcribing {describe_regions(pending)} in {format_chunk_size(size)} chunks…")
        units = _units_for(pending, size)
        failed: list[tuple[float, float]] = []
        level_errors: list[BaseException] = []

        for unit in units:
            try:
                done.extend(run_unit(unit))
            except Exception as exc:  # noqa: BLE001 — recorded, then retried smaller
                say(f"⚠️ {unit.label} failed: {exc}")
                failed.append((unit.start, unit.end))
                level_errors.append(exc)
                first_error = first_error or exc

        pending = merge_regions(failed)
        if not pending:
            return done

        # Every attempt so far failed the same way and nothing came out: this is
        # not a stall that smaller chunks can fix (missing model, unsupported
        # language, no engine), so stop instead of walking the whole ladder.
        # Checked from the second level on, since a whole-file failure that
        # chunking *does* fix (out of memory, one endless decode) looks the same
        # until a smaller chunk has been tried.
        if not done and level >= 1 and len({str(exc) for exc in level_errors}) == 1:
            raise level_errors[0]

        if level + 1 < len(levels):
            say(
                f"↩️ Retrying {describe_regions(pending)} in "
                f"{format_chunk_size(levels[level + 1] or 0)} chunks…"
            )

    if not done and first_error is not None:
        raise first_error
    message = (
        f"Could not transcribe {describe_regions(pending)} — it stalled at every "
        f"chunk size ({format_ladder(ladder)} min)."
    )
    if not skip_failed:
        raise StalledError(message)
    say(f"⚠️ Skipped {describe_regions(pending)} — stalled at every chunk size.")
    return done
