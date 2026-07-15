"""SBertEmbeddingModel lazy load must be thread-safe.

TreeBuilder.multithreaded_create_leaf_nodes races every worker into
``_ensure_model``. Unlocked, each one sees ``model is None`` and loads its own
438 MB SentenceTransformer -- 8 copies on a 4-CPU box, which OOMs a free-tier T4.
"""

import sys
import threading
import time
import types
from unittest import mock

from raptor.EmbeddingModels import SBertEmbeddingModel


def _fake_st_module(loads, delay=0.05):
    """A stand-in sentence_transformers whose ctor is slow enough to lose a race."""

    def _ctor(name):
        loads.append(name)
        time.sleep(delay)  # widen the window so an unlocked impl reliably double-loads
        return object()

    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = _ctor
    return mod


def test_ensure_model_loads_once_under_concurrent_threads():
    loads = []
    model = SBertEmbeddingModel()

    with mock.patch.dict(sys.modules, {"sentence_transformers": _fake_st_module(loads)}):
        threads = [threading.Thread(target=model._ensure_model) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert len(loads) == 1, f"loaded {len(loads)}x under 8 threads; expected 1"
    assert model.model is not None


def test_ensure_model_is_idempotent_and_returns_same_instance():
    loads = []
    model = SBertEmbeddingModel()

    with mock.patch.dict(sys.modules, {"sentence_transformers": _fake_st_module(loads)}):
        first = model._ensure_model()
        second = model._ensure_model()

    assert first is second
    assert len(loads) == 1


def test_unlocked_lazy_init_would_double_load():
    """Guard the guard: prove the test above can actually fail.

    Reproduces the pre-fix body (no lock) so that if someone drops the lock, the
    test above is known to catch it rather than passing vacuously.
    """
    loads = []
    holder = {"model": None}
    fake = _fake_st_module(loads)

    def _unlocked_ensure():
        if holder["model"] is None:
            holder["model"] = fake.SentenceTransformer("m")
        return holder["model"]

    threads = [threading.Thread(target=_unlocked_ensure) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(loads) > 1, "race did not reproduce; the locked test may be vacuous"
