import pytest

from raptor.SummarizationModels import BaseSummarizationModel
from raptor.QAModels import BaseQAModel
from experiments.config import ExperimentConfig
from experiments.models import (
    DiskCache,
    make_cache_key,
    CachedOpenRouterSummarizationModel,
    CachedOpenRouterQAModel,
    build_models,
    make_qa_model,
)


# ---------------------------------------------------------------------------
# Fake OpenAI-like client
# ---------------------------------------------------------------------------
class _Msg:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})()


class _Resp:
    def __init__(self, content):
        self.choices = [_Msg(content)]


class FakeClient:
    """Minimal stand-in for an OpenAI client. Records calls."""

    def __init__(self, content="CANNED RESPONSE", fail_times=0):
        self.content = content
        self.fail_times = fail_times
        self.calls = 0
        self.last_kwargs = None
        self.chat = self  # so .chat.completions.create works
        self.completions = self

    def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if self.calls <= self.fail_times:
            raise RuntimeError("simulated transient error (429-like)")
        return _Resp(self.content)


# ---------------------------------------------------------------------------
# DiskCache / make_cache_key
# ---------------------------------------------------------------------------
def test_disk_cache_round_trip(tmp_path):
    cache = DiskCache(str(tmp_path / "c"))
    assert cache.get("k") is None
    cache.set("k", "value")
    assert cache.get("k") == "value"


def test_make_cache_key_deterministic_and_sensitive():
    msgs = [{"role": "user", "content": "hi"}]
    k1 = make_cache_key("m1", msgs, max_tokens=10)
    k2 = make_cache_key("m1", msgs, max_tokens=10)
    assert k1 == k2
    # sensitive to model
    assert k1 != make_cache_key("m2", msgs, max_tokens=10)
    # sensitive to messages
    assert k1 != make_cache_key("m1", [{"role": "user", "content": "bye"}], max_tokens=10)
    # sensitive to extra kwargs
    assert k1 != make_cache_key("m1", msgs, max_tokens=99)


# ---------------------------------------------------------------------------
# Summarization model
# ---------------------------------------------------------------------------
def test_summarize_returns_content_and_is_base(tmp_path):
    client = FakeClient("a summary")
    cache = DiskCache(str(tmp_path / "c"))
    m = CachedOpenRouterSummarizationModel(client=client, cache=cache)
    assert isinstance(m, BaseSummarizationModel)
    out = m.summarize("some context")
    assert out == "a summary"
    assert client.calls == 1


def test_summarize_second_identical_call_served_from_cache(tmp_path):
    client = FakeClient("a summary")
    cache = DiskCache(str(tmp_path / "c"))
    m = CachedOpenRouterSummarizationModel(client=client, cache=cache)
    m.summarize("ctx")
    m.summarize("ctx")
    assert client.calls == 1  # second served from cache


def test_summarize_different_context_calls_again(tmp_path):
    client = FakeClient("a summary")
    cache = DiskCache(str(tmp_path / "c"))
    m = CachedOpenRouterSummarizationModel(client=client, cache=cache)
    m.summarize("ctx one")
    m.summarize("ctx two")
    assert client.calls == 2


def test_summarize_backoff_retries_then_succeeds(tmp_path):
    client = FakeClient("recovered", fail_times=1)
    cache = DiskCache(str(tmp_path / "c"))
    m = CachedOpenRouterSummarizationModel(client=client, cache=cache)
    out = m.summarize("ctx")
    assert out == "recovered"
    assert client.calls == 2  # failed once, succeeded on retry


# ---------------------------------------------------------------------------
# QA model
# ---------------------------------------------------------------------------
def test_answer_question_returns_content_and_is_base(tmp_path):
    client = FakeClient("the answer")
    cache = DiskCache(str(tmp_path / "c"))
    m = CachedOpenRouterQAModel(client=client, cache=cache)
    assert isinstance(m, BaseQAModel)
    out = m.answer_question("ctx", "what?")
    assert out == "the answer"
    assert client.calls == 1


def test_answer_question_cache_hit_on_repeat(tmp_path):
    client = FakeClient("the answer")
    cache = DiskCache(str(tmp_path / "c"))
    m = CachedOpenRouterQAModel(client=client, cache=cache)
    m.answer_question("ctx", "q")
    m.answer_question("ctx", "q")
    assert client.calls == 1


def test_answer_question_custom_prompts_are_used(tmp_path):
    client = FakeClient("the answer")
    cache = DiskCache(str(tmp_path / "c"))
    m = CachedOpenRouterQAModel(
        client=client,
        cache=cache,
        system_prompt="CUSTOM SYSTEM",
        user_template="CTX={context} || Q={question}",
    )
    m.answer_question("MYCTX", "MYQ")
    msgs = client.last_kwargs["messages"]
    assert msgs[0]["content"] == "CUSTOM SYSTEM"
    assert msgs[1]["content"] == "CTX=MYCTX || Q=MYQ"


def test_answer_question_backoff_retries_then_succeeds(tmp_path):
    client = FakeClient("recovered", fail_times=1)
    cache = DiskCache(str(tmp_path / "c"))
    m = CachedOpenRouterQAModel(client=client, cache=cache)
    out = m.answer_question("ctx", "q")
    assert out == "recovered"
    assert client.calls == 2


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------
def test_build_models_from_config(tmp_path):
    cfg = ExperimentConfig(model="x/y", cache_dir=str(tmp_path / "cache"), temperature=0.0)
    summ, qa = build_models(cfg)
    assert isinstance(summ, CachedOpenRouterSummarizationModel)
    assert isinstance(qa, CachedOpenRouterQAModel)
    assert summ.model == "x/y"
    assert qa.model == "x/y"
    # shared cache directory
    assert summ.cache.cache_dir == qa.cache.cache_dir


def test_make_qa_model_with_overrides(tmp_path):
    cfg = ExperimentConfig(model="x/y", cache_dir=str(tmp_path / "cache"))
    qa = make_qa_model(cfg, system_prompt="SYS", user_template="{context}-{question}")
    assert isinstance(qa, CachedOpenRouterQAModel)
    assert qa.system_prompt == "SYS"
    assert qa.user_template == "{context}-{question}"
