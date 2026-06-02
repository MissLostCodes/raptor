import hashlib
import json
import os
from typing import Callable, List, Optional


class JsonFileCache:
    """Simple one-file-per-key JSON cache on disk."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, key + ".json")

    def get(self, key: str):
        path = self._path(key)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def set(self, key: str, value) -> None:
        with open(self._path(key), "w", encoding="utf-8") as f:
            json.dump(value, f)


def make_key(model: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def cached_parse(
    text: str, model: str, cache: JsonFileCache, parse_fn: Callable[[str], List[str]]
) -> List[str]:
    key = make_key(model, text)
    hit = cache.get(key)
    if hit is not None:
        return hit
    result = parse_fn(text)
    cache.set(key, result)
    return result
