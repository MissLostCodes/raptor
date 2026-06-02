from raptor.chunking.llm_cache import JsonFileCache, make_key, cached_parse


def test_make_key_is_deterministic_and_model_sensitive():
    assert make_key("m1", "hello") == make_key("m1", "hello")
    assert make_key("m1", "hello") != make_key("m2", "hello")
    assert make_key("m1", "hello") != make_key("m1", "world")


def test_cache_round_trip(tmp_path):
    cache = JsonFileCache(str(tmp_path / "c"))
    assert cache.get("k") is None
    cache.set("k", ["a", "b"])
    assert cache.get("k") == ["a", "b"]


def test_cached_parse_calls_fn_once_then_serves_cache(tmp_path):
    cache = JsonFileCache(str(tmp_path / "c"))
    calls = {"n": 0}

    def parse_fn(text):
        calls["n"] += 1
        return ["T1", "T2"]

    first = cached_parse("doc", "modelX", cache, parse_fn)
    second = cached_parse("doc", "modelX", cache, parse_fn)
    assert first == ["T1", "T2"]
    assert second == ["T1", "T2"]
    assert calls["n"] == 1  # second call served from cache
