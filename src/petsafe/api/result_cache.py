"""Small process-local LRU cache for inference result payloads."""

import hashlib
import os
from collections import OrderedDict
from copy import deepcopy
from threading import RLock
from typing import Any

_MAX_ENTRIES = max(0, int(os.getenv("PETSAFE_RESULT_CACHE_SIZE", "256")))
_RESULTS: OrderedDict[str, dict[str, Any]] = OrderedDict()
_LOCK = RLock()


def image_digest(image_bytes: bytes) -> str:
    """Return a stable SHA-256 digest without retaining image bytes as a cache key."""
    return hashlib.sha256(image_bytes).hexdigest()


def get_cached_result(key: str) -> dict[str, Any] | None:
    """Return a copy of a cached payload and mark its key as recently used."""
    if _MAX_ENTRIES == 0:
        return None
    with _LOCK:
        result = _RESULTS.get(key)
        if result is None:
            return None
        _RESULTS.move_to_end(key)
        return deepcopy(result)


def cache_result(key: str, result: dict[str, Any]) -> None:
    """Store a copy of a JSON-compatible payload, evicting the least recent item."""
    if _MAX_ENTRIES == 0:
        return
    with _LOCK:
        _RESULTS[key] = deepcopy(result)
        _RESULTS.move_to_end(key)
        while len(_RESULTS) > _MAX_ENTRIES:
            _RESULTS.popitem(last=False)
