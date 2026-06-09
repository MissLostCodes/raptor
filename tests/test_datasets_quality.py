from experiments.datasets.quality import parse_quality_article


def _fixture():
    return {
        "article_id": "art1",
        "title": "An Article",
        "article": "Once upon a time there was a long story.",
        "questions": [
            {
                "question": "Who is the hero?",
                "options": ["Alice", "Bob", "Carol", "Dave"],
                "gold_label": 2,
                "difficult": 0,
            },
            {
                "question": "What is the theme?",
                "options": ["Love", "War", "Peace", "Hope"],
                "gold_label": 4,
                "difficult": 1,
            },
        ],
    }


def test_returns_document():
    doc = parse_quality_article(_fixture())
    assert doc.doc_id == "art1"
    assert doc.title == "An Article"
    assert "long story" in doc.text
    assert len(doc.questions) == 2


def test_gold_index_zero_based():
    doc = parse_quality_article(_fixture())
    assert doc.questions[0].gold_index == 1  # gold_label 2 -> idx 1
    assert doc.questions[1].gold_index == 3  # gold_label 4 -> idx 3


def test_is_hard_flag():
    doc = parse_quality_article(_fixture())
    assert doc.questions[0].is_hard is False
    assert doc.questions[1].is_hard is True


def test_options_preserved():
    doc = parse_quality_article(_fixture())
    for q in doc.questions:
        assert q.options is not None
        assert len(q.options) == 4


def test_missing_gold_label_gives_none():
    f = _fixture()
    del f["questions"][0]["gold_label"]
    doc = parse_quality_article(f)
    assert doc.questions[0].gold_index is None


def test_set_unique_id_fallback_for_doc_id():
    f = _fixture()
    del f["article_id"]
    f["set_unique_id"] = "set42"
    doc = parse_quality_article(f)
    assert doc.doc_id == "set42"


# --- field-name-variant robustness (different HF mirrors) -------------------
def test_gold_index_from_answer_field():
    # Mirror that uses 'answer' (1-based) instead of 'gold_label'.
    art = {
        "article_id": "x",
        "article": "body",
        "questions": [{"question": "q", "options": ["a", "b", "c", "d"], "answer": 3}],
    }
    doc = parse_quality_article(art)
    assert doc.questions[0].gold_index == 2  # 1-based 3 -> idx 2


def test_is_hard_from_hard_field():
    # Mirror that uses 'hard' instead of 'difficult'.
    art = {
        "article_id": "x",
        "article": "body",
        "questions": [{"question": "q", "options": ["a", "b"], "gold_label": 1, "hard": True}],
    }
    doc = parse_quality_article(art)
    assert doc.questions[0].is_hard is True


def test_hard_bool_not_mistaken_for_gold():
    # A bare 'hard' boolean must never be parsed as a gold label.
    art = {
        "article_id": "x",
        "article": "body",
        "questions": [{"question": "q", "options": ["a", "b"], "hard": True}],
    }
    doc = parse_quality_article(art)
    assert doc.questions[0].gold_index is None
    assert doc.questions[0].is_hard is True


def test_article_text_from_context_field():
    doc = parse_quality_article({"article_id": "x", "context": "the body", "questions": []})
    assert doc.text == "the body"
