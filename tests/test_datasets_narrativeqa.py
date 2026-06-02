from experiments.datasets.narrativeqa import group_narrativeqa_rows


def _rows():
    doc1 = {
        "id": "story1",
        "summary": {"text": "A short summary."},
        "text": "Full story one. It is quite long and html-ish.",
    }
    doc2 = {
        "id": "story2",
        "summary": {"text": "Another summary."},
        "text": "Full story two text.",
    }
    return [
        {
            "document": doc1,
            "question": {"text": "Q1 about story 1?"},
            "answers": [{"text": "ans1a"}, {"text": "ans1b"}],
        },
        {
            "document": doc1,
            "question": {"text": "Q2 about story 1?"},
            "answers": [{"text": "ans2a"}, {"text": "ans2b"}],
        },
        {
            "document": doc2,
            "question": {"text": "Q1 about story 2?"},
            "answers": [{"text": "ans3a"}, {"text": "ans3b"}],
        },
    ]


def test_groups_into_two_documents():
    docs = group_narrativeqa_rows(_rows())
    assert len(docs) == 2
    ids = {d.doc_id for d in docs}
    assert ids == {"story1", "story2"}


def test_question_counts_per_document():
    docs = {d.doc_id: d for d in group_narrativeqa_rows(_rows())}
    assert len(docs["story1"].questions) == 2
    assert len(docs["story2"].questions) == 1


def test_two_gold_answers_each():
    docs = {d.doc_id: d for d in group_narrativeqa_rows(_rows())}
    for q in docs["story1"].questions:
        assert len(q.gold_answers) == 2
    q = docs["story2"].questions[0]
    assert q.gold_answers == ["ans3a", "ans3b"]


def test_full_story_text_preserved():
    docs = {d.doc_id: d for d in group_narrativeqa_rows(_rows())}
    assert docs["story1"].text == "Full story one. It is quite long and html-ish."
    assert docs["story2"].text == "Full story two text."


def test_question_text_populated():
    docs = {d.doc_id: d for d in group_narrativeqa_rows(_rows())}
    qtexts = {q.question for q in docs["story1"].questions}
    assert qtexts == {"Q1 about story 1?", "Q2 about story 1?"}
