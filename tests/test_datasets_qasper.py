from experiments.datasets.qasper import parse_qasper_row


def _fixture_row():
    return {
        "id": "paper123",
        "title": "A Great NLP Paper",
        "abstract": "We study things.",
        "full_text": {
            "section_name": ["Introduction", "Methods"],
            "paragraphs": [
                ["Intro para one.", "Intro para two."],
                ["Methods para one."],
            ],
        },
        "qas": {
            "question": ["What is the F1?", "What is the capital of Mars?"],
            "question_id": ["q_ans", "q_unans"],
            "answers": [
                {
                    "answer": [
                        {
                            "unanswerable": False,
                            "extractive_spans": [],
                            "yes_no": None,
                            "free_form_answer": "The F1 is 0.9.",
                            "evidence": ["We report F1 of 0.9 in Methods."],
                        },
                        {
                            "unanswerable": False,
                            "extractive_spans": ["0.9 F1"],
                            "yes_no": None,
                            "free_form_answer": "",
                            "evidence": ["Table 1 shows 0.9 F1."],
                        },
                    ]
                },
                {
                    "answer": [
                        {
                            "unanswerable": True,
                            "extractive_spans": [],
                            "yes_no": None,
                            "free_form_answer": "",
                            "evidence": [],
                        }
                    ]
                },
            ],
        },
    }


def test_parse_returns_document_with_metadata():
    doc = parse_qasper_row(_fixture_row())
    assert doc.doc_id == "paper123"
    assert doc.title == "A Great NLP Paper"
    assert len(doc.questions) == 2


def test_text_contains_title_abstract_and_section_names():
    doc = parse_qasper_row(_fixture_row())
    assert "A Great NLP Paper" in doc.text
    assert "We study things." in doc.text
    assert "Introduction" in doc.text
    assert "Methods" in doc.text
    assert "Intro para one." in doc.text
    assert "Methods para one." in doc.text


def test_answerable_question_gold_answers_collected():
    doc = parse_qasper_row(_fixture_row())
    q = next(x for x in doc.questions if x.question_id == "q_ans")
    assert q.is_unanswerable is False
    # free_form from annotator 1 and extractive_spans from annotator 2
    assert "The F1 is 0.9." in q.gold_answers
    assert "0.9 F1" in q.gold_answers
    # evidence union
    assert "We report F1 of 0.9 in Methods." in q.evidence
    assert "Table 1 shows 0.9 F1." in q.evidence
    # annotator gold sets stored for evidence_f1_max
    assert "gold_evidence_per_annotator" in q.extra


def test_unanswerable_question_flag():
    doc = parse_qasper_row(_fixture_row())
    q = next(x for x in doc.questions if x.question_id == "q_unans")
    assert q.is_unanswerable is True
    assert q.gold_answers == []
    assert q.evidence == []


def test_yes_no_answer_mapped():
    row = _fixture_row()
    row["qas"]["answers"][0]["answer"][0] = {
        "unanswerable": False,
        "extractive_spans": [],
        "yes_no": True,
        "free_form_answer": "",
        "evidence": [],
    }
    doc = parse_qasper_row(row)
    q = next(x for x in doc.questions if x.question_id == "q_ans")
    assert "Yes" in q.gold_answers


def test_list_of_dicts_full_text_schema_supported():
    row = _fixture_row()
    row["full_text"] = [
        {"section_name": "Introduction", "paragraphs": ["Intro para one."]},
        {"section_name": "Methods", "paragraphs": ["Methods para one."]},
    ]
    doc = parse_qasper_row(row)
    assert "Introduction" in doc.text
    assert "Methods para one." in doc.text
