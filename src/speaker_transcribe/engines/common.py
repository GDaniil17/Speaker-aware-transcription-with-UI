"""Pieces every engine shares: progress callbacks and the loaded-model cache."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable

ProgressFn = Callable[[str], None]
ProgressBarFn = Callable[[str, float, float | None], None]


def log(progress: ProgressFn | None, msg: str) -> None:
    if progress:
        progress(msg)


# The loaded model, keyed by the settings that shape it. A chunked run calls the
# engine once per chunk, and reloading a large model every time would cost more
# than transcribing does. Only one entry is kept, so a model is never held in
# memory next to its replacement.
_MODEL_CACHE: dict[str, object] = {}


def cached_model(key: tuple, build: Callable[[], object]):
    """Return the model for `key`, reusing the previous one when it matches."""
    if _MODEL_CACHE.get("key") != key:
        _MODEL_CACHE.clear()
        _MODEL_CACHE["model"] = build()
        _MODEL_CACHE["key"] = key
    return _MODEL_CACHE["model"]


def module_available(name: str) -> bool:
    """True when a module can be imported (used to pick optional engine backends)."""
    return importlib.util.find_spec(name) is not None
