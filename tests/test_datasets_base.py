from experiments.datasets.base import (
    QAExample,
    Document,
    DatasetLoader,
    select_subset,
)


def test_qaexample_defaults():
    ex = QAExample(question_id="q1", question="What?")
    assert ex.gold_answers == []
    assert ex.options is None
    assert ex.gold_index is None
    assert ex.evidence == []
    assert ex.is_unanswerable is False
    assert ex.is_hard is False
    assert ex.extra == {}
    # Independent mutable defaults
    ex2 = QAExample(question_id="q2", question="Why?")
    ex.gold_answers.append("a")
    assert ex2.gold_answers == []


def test_document_defaults():
    doc = Document(doc_id="d1", title="T", text="body")
    assert doc.questions == []
    assert doc.meta == {}
    doc2 = Document(doc_id="d2", title="T2", text="body2")
    doc.questions.append(QAExample(question_id="q", question="?"))
    assert doc2.questions == []


def _docs(n):
    return [Document(doc_id=f"d{i}", title=f"t{i}", text="x") for i in range(n)]


def test_select_subset_deterministic_same_seed():
    docs = _docs(20)
    a = select_subset(docs, 5, seed=0)
    b = select_subset(docs, 5, seed=0)
    assert a == b
    assert len(a) == 5
    assert len(set(a)) == 5  # no duplicates


def test_select_subset_different_seed_can_differ():
    docs = _docs(50)
    a = select_subset(docs, 10, seed=0)
    b = select_subset(docs, 10, seed=12345)
    assert a != b


def test_select_subset_respects_n_capped_at_len():
    docs = _docs(3)
    out = select_subset(docs, 10, seed=0)
    assert len(out) == 3
    assert sorted(out) == ["d0", "d1", "d2"]


def test_select_subset_ids_are_real_doc_ids():
    docs = _docs(8)
    out = select_subset(docs, 4, seed=1)
    ids = {d.doc_id for d in docs}
    assert all(o in ids for o in out)


def test_select_subset_stratified_spans_strata():
    # 10 docs of stratum "A", 10 of "B"
    docs = []
    for i in range(10):
        docs.append(Document(doc_id=f"a{i}", title="t", text="x", meta={"s": "A"}))
    for i in range(10):
        docs.append(Document(doc_id=f"b{i}", title="t", text="x", meta={"s": "B"}))
    out = select_subset(docs, 10, seed=0, strata_fn=lambda d: d.meta["s"])
    assert len(out) == 10
    a_count = sum(1 for o in out if o.startswith("a"))
    b_count = sum(1 for o in out if o.startswith("b"))
    assert a_count > 0 and b_count > 0
    # proportional 50/50 -> 5 each
    assert a_count == 5 and b_count == 5


def test_select_subset_stratified_deterministic():
    docs = []
    for i in range(6):
        docs.append(Document(doc_id=f"a{i}", title="t", text="x", meta={"s": "A"}))
    for i in range(4):
        docs.append(Document(doc_id=f"b{i}", title="t", text="x", meta={"s": "B"}))
    fn = lambda d: d.meta["s"]
    a = select_subset(docs, 5, seed=7, strata_fn=fn)
    b = select_subset(docs, 5, seed=7, strata_fn=fn)
    assert a == b


def test_dataset_loader_is_abstract():
    import abc
    assert issubclass(DatasetLoader, abc.ABC)
    try:
        DatasetLoader()  # type: ignore
        assert False, "should not instantiate abstract base"
    except TypeError:
        pass
