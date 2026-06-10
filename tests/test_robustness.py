"""Offline tests for the moderation/robustness hardening of the harness.

Covers three concerns that combined to kill a 1-hour Colab run on a single
provider moderation false-positive:

1. models.py: a moderation 403 becomes a non-retryable ``ModerationBlocked``
   (NOT retried 6x), other 4xx are non-retryable, 429/5xx stay retryable, and
   summarization degrades gracefully instead of crashing the tree build.
2. runner.py: a blocked/errored question is recorded and the grid keeps going;
   runs are resumable (completed (dataset, arm, doc) are skipped).
3. report.py: a question blocked for ANY arm is dropped from ALL arms so the
   arm-vs-arm comparison stays fair.
"""

import json
import os

import pytest

from experiments.config import ExperimentConfig
from experiments.datasets.base import Document, QAExample
from experiments import runner, report
from experiments.models import (
    DiskCache,
    CachedOpenRouterSummarizationModel,
    CachedOpenRouterQAModel,
    ModerationBlocked,
)


# ---------------------------------------------------------------------------
# Fake OpenAI-like client that can raise a specific exception
# ---------------------------------------------------------------------------
class _Msg:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})()


class _Resp:
    def __init__(self, content):
        self.choices = [_Msg(content)]


class _StatusError(Exception):
    """Mimics an openai.APIStatusError: carries ``status_code`` + ``body``."""

    def __init__(self, status_code, message="", body=None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.body = body


def _moderation_403():
    return _StatusError(
        403,
        message=(
            "openai/gpt-oss-120b:free requires moderation on OpenInference. "
            'Your input was flagged for "sexual/minors". No credits were charged.'
        ),
        body={"metadata": {"reasons": ["sexual/minors"]}},
    )


class FakeClient:
    def __init__(self, content="CANNED", fail_times=0, raise_exc=None):
        self.content = content
        self.fail_times = fail_times
        self.raise_exc = raise_exc
        self.calls = 0
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.raise_exc or RuntimeError("transient")
        return _Resp(self.content)


# ---------------------------------------------------------------------------
# models: retry classification
# ---------------------------------------------------------------------------
def test_moderation_403_raises_moderation_blocked_and_is_not_retried(tmp_path):
    client = FakeClient(fail_times=99, raise_exc=_moderation_403())
    m = CachedOpenRouterQAModel(client=client, cache=DiskCache(str(tmp_path / "c")))
    with pytest.raises(ModerationBlocked) as ei:
        m.answer_question("ctx", "q")
    # Permanent error: attempted exactly once, not 6x.
    assert client.calls == 1
    assert "sexual/minors" in str(ei.value)


def test_permanent_4xx_is_not_retried(tmp_path):
    client = FakeClient(fail_times=99, raise_exc=_StatusError(400, "bad request"))
    m = CachedOpenRouterQAModel(client=client, cache=DiskCache(str(tmp_path / "c")))
    with pytest.raises(_StatusError):
        m.answer_question("ctx", "q")
    assert client.calls == 1


def test_rate_limit_429_is_retried_then_succeeds(tmp_path):
    client = FakeClient(content="ok", fail_times=1, raise_exc=_StatusError(429, "rate"))
    m = CachedOpenRouterQAModel(client=client, cache=DiskCache(str(tmp_path / "c")))
    assert m.answer_question("ctx", "q") == "ok"
    assert client.calls == 2


def test_server_5xx_is_retried_then_succeeds(tmp_path):
    client = FakeClient(content="ok", fail_times=1, raise_exc=_StatusError(503, "down"))
    m = CachedOpenRouterQAModel(client=client, cache=DiskCache(str(tmp_path / "c")))
    assert m.answer_question("ctx", "q") == "ok"
    assert client.calls == 2


def test_summarize_moderation_falls_back_to_extractive(tmp_path):
    context = "ALPHA beta gamma delta " * 50
    client = FakeClient(fail_times=99, raise_exc=_moderation_403())
    m = CachedOpenRouterSummarizationModel(client=client, cache=DiskCache(str(tmp_path / "c")))
    out = m.summarize(context, max_tokens=10)
    # No raise: tree-building survives. Fallback is an extract of the context.
    assert out
    assert out in context or context.startswith(out)
    assert client.calls == 1  # not retried


def test_none_content_is_coerced_to_empty_string_not_crash(tmp_path):
    # gpt-oss is a reasoning model; its OpenAI-compatible response can carry
    # message.content=None (e.g. the token budget was spent on reasoning). The
    # harness MUST coerce that to a string so it never reaches the tree builder's
    # tiktoken.encode(None), which raises "expected string or buffer" and killed
    # the whole tree build (every doc skipped -> 0 records).
    client = FakeClient(content=None)
    s = CachedOpenRouterSummarizationModel(client=client, cache=DiskCache(str(tmp_path / "cs")))
    assert s.summarize("some context") == ""

    qa = CachedOpenRouterQAModel(client=client, cache=DiskCache(str(tmp_path / "cq")))
    assert qa.answer_question("ctx", "q") == ""


# ---------------------------------------------------------------------------
# runner: per-question isolation + resume
# ---------------------------------------------------------------------------
def _doc():
    return Document(
        doc_id="d1",
        title="T",
        text="body",
        questions=[
            QAExample(question_id="q1", question="?", options=["a", "b", "c", "d"], gold_index=2),
            QAExample(question_id="q2", question="?", options=["a", "b", "c", "d"], gold_index=0),
        ],
    )


class FakeLoader:
    def __init__(self, doc):
        self._doc = doc

    def load(self, subset_ids=None, limit=None):
        return [self._doc]


def _make_build(block_arm_qid=None, counter=None):
    """Build fn whose answerer raises ModerationBlocked for one (arm) only.

    block_arm_qid: ("ahc",) blocks the FIRST question for arm 'ahc'.
    """

    def build(arm, doc, cfg, summ, qa):
        if counter is not None:
            counter.append((arm, doc.doc_id))
        block = block_arm_qid is not None and arm == block_arm_qid

        class A:
            def __init__(self):
                self._n = 0

            def answer(self, question):
                self._n += 1
                if block and self._n == 1:
                    raise ModerationBlocked("flagged: sexual/minors")
                return "C", "ctx"

            def routing_info(self):
                return {}

        return A()

    return build


def test_blocked_question_is_recorded_and_run_continues(tmp_path):
    cfg = ExperimentConfig(
        arms=["token", "ahc"],
        datasets=["quality"],
        subset_sizes={"quality": 1},
        results_dir=str(tmp_path),
    )
    result = runner.run(
        cfg,
        build_answerer_fn=_make_build(block_arm_qid="ahc"),
        loaders={"quality": FakeLoader(_doc())},
        seed=0,
        progress=lambda *a, **k: None,
        show_progress=False,
    )
    recs = result["records"]
    # 2 arms * 2 questions = 4 records, none lost despite the block.
    assert len(recs) == 4
    blocked = [r for r in recs if r.get("blocked")]
    assert len(blocked) == 1
    assert blocked[0]["arm"] == "ahc"
    assert blocked[0]["error"] == "moderation"
    # The blocked record carries no score field that would pollute aggregation.
    assert "correct" not in blocked[0]
    # The same question answered fine under 'token'.
    tok = [r for r in recs if r["arm"] == "token"]
    assert all(r["blocked"] is False for r in tok)


def test_build_failure_skips_doc_and_continues(tmp_path):
    def build(arm, doc, cfg, summ, qa):
        if arm == "ahc":
            raise RuntimeError("tree build exploded")

        class A:
            def answer(self, q):
                return "C", "ctx"

            def routing_info(self):
                return {}

        return A()

    cfg = ExperimentConfig(
        arms=["token", "ahc"],
        datasets=["quality"],
        subset_sizes={"quality": 1},
        results_dir=str(tmp_path),
    )
    result = runner.run(
        cfg,
        build_answerer_fn=build,
        loaders={"quality": FakeLoader(_doc())},
        seed=0,
        progress=lambda *a, **k: None,
        show_progress=False,
    )
    recs = result["records"]
    # token doc fully recorded (2 q); ahc doc skipped (build failed) => 2 records.
    assert {r["arm"] for r in recs} == {"token"}
    assert len(recs) == 2


def test_run_is_resumable_skips_completed_docs(tmp_path):
    counter = []
    cfg = ExperimentConfig(
        arms=["token", "ahc"],
        datasets=["quality"],
        subset_sizes={"quality": 1},
        results_dir=str(tmp_path),
    )
    common = dict(
        loaders={"quality": FakeLoader(_doc())},
        seed=0,
        progress=lambda *a, **k: None,
        show_progress=False,
    )
    runner.run(cfg, build_answerer_fn=_make_build(counter=counter), **common)
    first_pass_builds = len(counter)
    assert first_pass_builds == 2  # token+ahc each built once

    # Second run with resume should skip both completed (dataset, arm, doc).
    result2 = runner.run(
        cfg, build_answerer_fn=_make_build(counter=counter), resume=True, **common
    )
    assert len(counter) == first_pass_builds  # no new builds
    assert len(result2["records"]) == 4  # still complete


# ---------------------------------------------------------------------------
# report: uniform exclusion of blocked questions
# ---------------------------------------------------------------------------
def test_exclude_blocked_uniformly_drops_question_from_all_arms():
    records = [
        # q1 blocked for ahc, answered for token.
        {"arm": "token", "dataset": "quality", "doc_id": "d1", "question_id": "q1",
         "correct": 1, "is_hard": False, "blocked": False, "error": None, "routing": {}},
        {"arm": "ahc", "dataset": "quality", "doc_id": "d1", "question_id": "q1",
         "blocked": True, "error": "moderation", "routing": {}},
        # q2 answered for both.
        {"arm": "token", "dataset": "quality", "doc_id": "d1", "question_id": "q2",
         "correct": 0, "is_hard": False, "blocked": False, "error": None, "routing": {}},
        {"arm": "ahc", "dataset": "quality", "doc_id": "d1", "question_id": "q2",
         "correct": 1, "is_hard": False, "blocked": False, "error": None, "routing": {}},
    ]
    kept = report.exclude_blocked_uniformly(records)
    # q1 removed from BOTH arms; only q2 remains for each arm.
    qids = {(r["arm"], r["question_id"]) for r in kept}
    assert qids == {("token", "q2"), ("ahc", "q2")}

    agg = report.aggregate(records)  # aggregate applies the uniform drop
    assert agg["token"]["quality"]["accuracy"]["n"] == 1
    assert agg["ahc"]["quality"]["accuracy"]["n"] == 1

    rep = report.blocked_report(records)
    assert rep["quality"]["dropped_questions"] == 1
