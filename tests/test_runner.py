"""Offline tests for experiments.runner (prompts, scorers, orchestration loop)."""

import json
import os

from experiments.config import ExperimentConfig
from experiments.datasets.base import Document, QAExample
from experiments import runner


# ---------------------------------------------------------------------------
# qa_prompt_for / format_options
# ---------------------------------------------------------------------------
def test_qa_prompt_for_three_distinct_templates():
    templates = {
        runner.qa_prompt_for(name)[1]
        for name in ("qasper", "quality", "narrativeqa")
    }
    assert len(templates) == 3
    # Templates only use the two fields the QA model fills; options are folded
    # into the question by compose_query, not via an unfillable placeholder.
    for name in ("qasper", "quality", "narrativeqa"):
        tmpl = runner.qa_prompt_for(name)[1]
        assert "{options}" not in tmpl
        assert "{context}" in tmpl and "{question}" in tmpl
    # The quality template still instructs the model to answer with a letter.
    assert "letter" in runner.qa_prompt_for("quality")[1].lower()


def test_format_options():
    out = runner.format_options(["cat", "dog", "bird"])
    assert out == "A) cat\nB) dog\nC) bird"


def test_compose_query_folds_quality_options_into_question():
    ex = QAExample(
        question_id="q1",
        question="What color is the sky?",
        options=["red", "blue", "green", "black"],
        gold_index=1,
    )
    q = runner.compose_query("quality", ex)
    assert "What color is the sky?" in q
    assert "A) red" in q and "B) blue" in q
    # Non-quality datasets pass the bare question through unchanged.
    ex2 = QAExample(question_id="q2", question="Who?", gold_answers=["x"])
    assert runner.compose_query("qasper", ex2) == "Who?"


# ---------------------------------------------------------------------------
# score_question per dataset
# ---------------------------------------------------------------------------
def test_score_qasper_exact_match_f1_is_one():
    ex = QAExample(question_id="q1", question="?", gold_answers=["the answer"])
    rec = runner.score_question("qasper", ex, "the answer")
    assert rec["answer_f1"] == 1.0
    assert rec["evidence_f1"] is None


def test_score_qasper_evidence_coverage_from_context():
    ex = QAExample(
        question_id="q1",
        question="?",
        gold_answers=["x"],
        evidence=["alpha beta gamma"],
    )
    rec = runner.score_question(
        "qasper", ex, "x", retrieved_context="prefix alpha beta gamma suffix"
    )
    assert rec["evidence_coverage"] == 1.0
    assert rec["evidence_f1"] is None  # paragraph-set F1 inapplicable here


def test_score_qasper_evidence_coverage_none_without_gold_evidence():
    ex = QAExample(question_id="q1", question="?", gold_answers=["x"])
    rec = runner.score_question("qasper", ex, "x", retrieved_context="anything")
    assert rec["evidence_coverage"] is None


def test_score_qasper_unanswerable_abstain():
    ex = QAExample(question_id="q1", question="?", gold_answers=[], is_unanswerable=True)
    rec = runner.score_question("qasper", ex, "Unanswerable")
    assert rec["answer_f1"] == 1.0


def test_score_quality_correct_and_hard():
    ex = QAExample(
        question_id="q1",
        question="?",
        options=["a", "b", "c", "d"],
        gold_index=2,
        is_hard=True,
    )
    rec = runner.score_question("quality", ex, "C")
    assert rec["choice"] == 2
    assert rec["correct"] == 1
    assert rec["is_hard"] is True


def test_score_quality_incorrect():
    ex = QAExample(
        question_id="q1",
        question="?",
        options=["a", "b", "c", "d"],
        gold_index=0,
        is_hard=False,
    )
    rec = runner.score_question("quality", ex, "D")
    assert rec["correct"] == 0
    assert rec["is_hard"] is False


def test_score_narrativeqa_returns_metric_dict():
    ex = QAExample(question_id="q1", question="?", gold_answers=["a small dog"])
    rec = runner.score_question("narrativeqa", ex, "a small dog")
    assert set(rec.keys()) == {"rouge_l", "bleu_1", "bleu_4", "meteor"}
    assert rec["rouge_l"] == 1.0 or rec["rouge_l"] > 0.9


# ---------------------------------------------------------------------------
# run() orchestration with fakes
# ---------------------------------------------------------------------------
class FakeLoader:
    def __init__(self, doc):
        self._doc = doc

    def load(self, subset_ids=None, limit=None):
        return [self._doc]


class FakeAnswerer:
    def __init__(self, arm):
        self._arm = arm

    def answer(self, question):
        return "C", "some retrieved context"

    def routing_info(self):
        if self._arm == "ahc":
            return {"route": "structure", "score": 0.9}
        return {}


def _fake_build_answerer(arm, doc, cfg, summ_model, qa_model):
    return FakeAnswerer(arm)


class RecordingLoader:
    """Fake loader that records how it was called (subset_ids vs limit)."""

    def __init__(self, doc):
        self._doc = doc
        self.calls = []

    def load(self, subset_ids=None, limit=None):
        self.calls.append({"subset_ids": subset_ids, "limit": limit})
        return [self._doc]


def _one_quality_doc():
    return Document(
        doc_id="d1",
        title="T",
        text="body",
        questions=[
            QAExample(
                question_id="q1",
                question="?",
                options=["a", "b", "c", "d"],
                gold_index=2,
            )
        ],
    )


def test_run_raises_on_unpinned_subset(tmp_path):
    # No pin file in subsets_dir and allow_unpinned not set -> hard error, so a
    # real run can never silently evaluate the wrong (unreproducible) documents.
    cfg = ExperimentConfig(
        arms=["token"],
        datasets=["quality"],
        subset_sizes={"quality": 1},
        results_dir=str(tmp_path / "results"),
    )
    loader = RecordingLoader(_one_quality_doc())
    try:
        runner.run(
            cfg,
            build_answerer_fn=_fake_build_answerer,
            loaders={"quality": loader},
            seed=0,
            progress=lambda *a, **k: None,
            subsets_dir=str(tmp_path / "subsets"),
        )
        assert False, "expected RuntimeError for unpinned subset"
    except RuntimeError as exc:
        assert "quality" in str(exc)
        assert "select_subsets" in str(exc)
    # The loader must NOT have been asked to load anything.
    assert loader.calls == []


def test_run_uses_pinned_ids_when_present(tmp_path):
    subsets_dir = tmp_path / "subsets"
    subsets_dir.mkdir()
    (subsets_dir / "quality_ids.json").write_text(
        json.dumps({"seed": 0, "ids": ["d1"]})
    )
    cfg = ExperimentConfig(
        arms=["token"],
        datasets=["quality"],
        subset_sizes={"quality": 1},
        results_dir=str(tmp_path / "results"),
    )
    loader = RecordingLoader(_one_quality_doc())
    runner.run(
        cfg,
        build_answerer_fn=_fake_build_answerer,
        loaders={"quality": loader},
        seed=0,
        progress=lambda *a, **k: None,
        subsets_dir=str(subsets_dir),
    )
    # Loaded by pinned ids, NOT by limit.
    assert loader.calls == [{"subset_ids": ["d1"], "limit": None}]


def test_run_allow_unpinned_falls_back_to_limit(tmp_path):
    cfg = ExperimentConfig(
        arms=["token"],
        datasets=["quality"],
        subset_sizes={"quality": 1},
        results_dir=str(tmp_path / "results"),
    )
    loader = RecordingLoader(_one_quality_doc())
    runner.run(
        cfg,
        build_answerer_fn=_fake_build_answerer,
        loaders={"quality": loader},
        seed=0,
        progress=lambda *a, **k: None,
        subsets_dir=str(tmp_path / "subsets"),
        allow_unpinned=True,
    )
    # Explicit opt-out: falls back to first-N-by-load-order.
    assert loader.calls == [{"subset_ids": None, "limit": 1}]


def test_run_threads_retrieved_context_into_qasper_evidence_coverage(tmp_path):
    doc = Document(
        doc_id="p1",
        title="T",
        text="body",
        questions=[
            QAExample(
                question_id="q1",
                question="?",
                gold_answers=["x"],
                evidence=["alpha beta gamma"],
            )
        ],
    )

    class CtxAnswerer:
        def answer(self, q):
            return "x", "prefix alpha beta gamma suffix"

        def routing_info(self):
            return {}

    cfg = ExperimentConfig(
        arms=["token"],
        datasets=["qasper"],
        subset_sizes={"qasper": 1},
        results_dir=str(tmp_path),
    )
    result = runner.run(
        cfg,
        build_answerer_fn=lambda *a: CtxAnswerer(),
        loaders={"qasper": FakeLoader(doc)},
        seed=0,
        progress=lambda *a, **k: None,
        allow_unpinned=True,
    )
    rec = result["records"][0]
    # The retrieved context surfaced all gold-evidence tokens -> coverage 1.0,
    # proving the runner threads context into scoring.
    assert rec["evidence_coverage"] == 1.0


def test_run_records_build_and_answer_timings(tmp_path):
    doc = Document(
        doc_id="d1",
        title="T",
        text="body",
        questions=[
            QAExample(question_id="q1", question="?", options=["a", "b", "c", "d"], gold_index=2),
            QAExample(question_id="q2", question="?", options=["a", "b", "c", "d"], gold_index=0),
        ],
    )
    ticks = iter(range(1000))
    clock = lambda: float(next(ticks))  # noqa: E731 - deterministic fake clock
    cfg = ExperimentConfig(
        arms=["token"],
        datasets=["quality"],
        subset_sizes={"quality": 1},
        results_dir=str(tmp_path),
    )
    result = runner.run(
        cfg,
        build_answerer_fn=_fake_build_answerer,
        loaders={"quality": FakeLoader(doc)},
        seed=0,
        progress=lambda *a, **k: None,
        allow_unpinned=True,
        clock=clock,
    )
    for r in result["records"]:
        assert r["build_s"] >= 0.0
        assert r["answer_s"] >= 0.0
    # Both questions of the doc share the same build_s (the doc is built once).
    assert len({r["build_s"] for r in result["records"]}) == 1


def test_run_produces_one_record_per_arm_dataset_question(tmp_path):
    doc = Document(
        doc_id="d1",
        title="T",
        text="body",
        questions=[
            QAExample(
                question_id="q1",
                question="?",
                options=["a", "b", "c", "d"],
                gold_index=2,
                is_hard=True,
            ),
            QAExample(
                question_id="q2",
                question="?",
                options=["a", "b", "c", "d"],
                gold_index=0,
            ),
        ],
    )
    cfg = ExperimentConfig(
        arms=["token", "ahc"],
        datasets=["quality"],
        subset_sizes={"quality": 1},
        results_dir=str(tmp_path),
    )
    result = runner.run(
        cfg,
        build_answerer_fn=_fake_build_answerer,
        loaders={"quality": FakeLoader(doc)},
        seed=0,
        progress=lambda *a, **k: None,
        allow_unpinned=True,  # orchestration test; pinning is covered separately
    )
    records = result["records"]
    # 2 arms * 1 dataset * 1 doc * 2 questions = 4 records.
    assert len(records) == 4
    arms = {r["arm"] for r in records}
    assert arms == {"token", "ahc"}

    # routing captured for ahc only.
    ahc_recs = [r for r in records if r["arm"] == "ahc"]
    assert all(r["routing"].get("route") == "structure" for r in ahc_recs)
    token_recs = [r for r in records if r["arm"] == "token"]
    assert all(r["routing"] == {} for r in token_recs)

    # scoring threaded through.
    q1 = next(r for r in records if r["question_id"] == "q1" and r["arm"] == "token")
    assert q1["correct"] == 1  # answered "C" == gold_index 2

    # JSON written.
    out_path = os.path.join(str(tmp_path), "results_0.json")
    assert os.path.exists(out_path)
    with open(out_path) as fh:
        on_disk = json.load(fh)
    assert len(on_disk["records"]) == 4
    assert "config" in on_disk
