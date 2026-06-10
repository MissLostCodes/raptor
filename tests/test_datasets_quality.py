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
    # 'answer' is the index-style field (0-based) used by emozilla/quality, where
    # values range over 0..3 (verified: 491 rows have answer==0). It must be used
    # as-is, NOT decremented like the 1-based 'gold_label'.
    art = {
        "article_id": "x",
        "article": "body",
        "questions": [{"question": "q", "options": ["a", "b", "c", "d"], "answer": 3}],
    }
    doc = parse_quality_article(art)
    assert doc.questions[0].gold_index == 3  # 0-based 3 -> idx 3


def test_gold_index_answer_zero_is_valid_not_negative():
    # The off-by-one bug mapped answer==0 -> -1 (always scored wrong) for ~24% of
    # QuALITY. 0-based answer==0 must map to idx 0.
    art = {
        "article_id": "x",
        "article": "body",
        "questions": [{"question": "q", "options": ["a", "b", "c", "d"], "answer": 0}],
    }
    doc = parse_quality_article(art)
    assert doc.questions[0].gold_index == 0


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


# --- one-row-per-question mirror (emozilla/quality) -------------------------
def test_normalize_groups_one_row_per_question_by_article():
    # emozilla/quality: one row per question, NO article id, 0-based 'answer'.
    # Without grouping-by-article the loader collapsed all 2086 questions into a
    # single doc_id='' Document with one wrong article -> meaningless results.
    from experiments.datasets.quality import QuALITYLoader

    rows = [
        {"article": "STORY ALPHA body", "question": "qa1",
         "options": ["a", "b", "c", "d"], "answer": 0, "hard": False},
        {"article": "STORY ALPHA body", "question": "qa2",
         "options": ["a", "b", "c", "d"], "answer": 3, "hard": True},
        {"article": "STORY BETA body", "question": "qb1",
         "options": ["a", "b", "c", "d"], "answer": 2, "hard": False},
    ]
    docs = QuALITYLoader._normalize(rows)

    # Two distinct articles -> two Documents (NOT collapsed into one).
    assert len(docs) == 2
    alpha = next(d for d in docs if d.text == "STORY ALPHA body")
    beta = next(d for d in docs if d.text == "STORY BETA body")
    assert len(alpha.questions) == 2
    assert len(beta.questions) == 1

    # Non-empty, distinct doc ids (no more doc_id='').
    assert alpha.doc_id and beta.doc_id and alpha.doc_id != beta.doc_id

    # 0-based answer used as-is; hard flag preserved.
    assert alpha.questions[0].gold_index == 0
    assert alpha.questions[1].gold_index == 3
    assert alpha.questions[1].is_hard is True

    # Question ids are non-empty and unique within a document.
    qids = [q.question_id for q in alpha.questions]
    assert all(qids) and len(set(qids)) == 2
