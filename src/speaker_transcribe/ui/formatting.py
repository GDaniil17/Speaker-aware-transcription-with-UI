"""Durations as the interface shows them."""

from __future__ import annotations

from speaker_transcribe.ui.i18n import is_russian


def format_duration(seconds: float) -> str:
    """Format a duration as a clock, e.g. '00:42' or '1:05:03'."""
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def format_step_duration(seconds: float) -> str:
    """Keep short steps visible instead of rounding them down to 00:00."""
    russian = is_russian()
    if seconds < 0.05:
        return "<0,1 с" if russian else "<0.1 s"
    if seconds < 60:
        amount = f"{max(seconds, 0):.1f}"
        return f"{amount.replace('.', ',')} с" if russian else f"{amount} s"
    return format_duration(seconds)


def format_total_duration(seconds: float) -> str:
    """Show a wall-clock total with units, using minutes for longer runs."""
    if round(seconds, 1) < 60:
        return format_step_duration(seconds)
    hours, remainder = divmod(max(0, int(round(seconds))), 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    hour_unit, minute_unit, second_unit = ("ч", "мин", "с") if is_russian() else ("h", "min", "s")
    parts = []
    if hours:
        parts.append(f"{hours} {hour_unit}")
    if minutes:
        parts.append(f"{minutes} {minute_unit}")
    if remaining_seconds:
        parts.append(f"{remaining_seconds} {second_unit}")
    return " ".join(parts)
