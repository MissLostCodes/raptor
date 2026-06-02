"""QASPER loader (NLP-paper QA; Answer-F1 + Evidence-F1).

Assumed HuggingFace schema (``allenai/qasper``), one row per paper::

    {
      "id":       str,                       # paper id
      "title":    str,
      "abstract": str,
      "full_text": {                         # dict of two PARALLEL lists
          "section_name": [str, ...],
          "paragraphs":   [[str, ...], ...], # paragraphs[i] -> section i
      },
      "qas": {                               # dict of PARALLEL lists
          "question":    [str, ...],
          "question_id": [str, ...],
          "answers":     [ {"answer": [ {                 # per annotator
                  "unanswerable":     bool,
                  "extractive_spans": [str, ...],
                  "yes_no":           bool | None,
                  "free_form_answer": str,
                  "evidence":         [str, ...],
              }, ... ]}, ... ],
      },
    }

``parse_qasper_row`` also tolerates the alternative ``full_text`` shape of a
list of ``{"section_name": str, "paragraphs": [str, ...]}`` dicts.

For each question we collect the non-empty gold answer string from every
annotator (free_form_answer; else "Yes"/"No" from yes_no; else the joined
extractive_spans), mark the question unanswerable iff every annotation is
unanswerable, take the union of evidence paragraphs, and stash the per-
annotator evidence sets in ``extra["gold_evidence_per_annotator"]`` so that
``evidence_f1_max`` can score against the best-matching annotator.
"""

from __future__ import annotations

from typing import List, Optional

from experiments.datasets.base import DatasetLoader, Document, QAExample


def _iter_sections(full_text):
    """Yield (section_name, paragraphs_list) over either supported schema."""
    if isinstance(full_text, dict):
        names = full_text.get("section_name", []) or []
        paras = full_text.get("paragraphs", []) or []
        for i in range(max(len(names), len(paras))):
            name = names[i] if i < len(names) else ""
            ps = paras[i] if i < len(paras) else []
            yield name, ps
    elif isinstance(full_text, list):
        for sec in full_text:
            yield sec.get("section_name", ""), sec.get("paragraphs", []) or []


def _build_text(row: dict) -> str:
    parts: List[str] = []
    title = row.get("title", "") or ""
    abstract = row.get("abstract", "") or ""
    if title:
        parts.append(title)
    if abstract:
        parts.append(abstract)
    for name, paras in _iter_sections(row.get("full_text")):
        section_block = ""
        if name:
            section_block += f"{name}\n"
        if paras:
            section_block += "\n\n".join(p for p in paras if p)
        if section_block.strip():
            parts.append(section_block)
    return "\n\n".join(parts)


def _answer_string(ans: dict) -> Optional[str]:
    """Extract the gold answer string for one annotator answer dict."""
    if ans.get("unanswerable"):
        return None
    free = (ans.get("free_form_answer") or "").strip()
    if free:
        return free
    yn = ans.get("yes_no")
    if yn is True:
        return "Yes"
    if yn is False:
        return "No"
    spans = [s for s in (ans.get("extractive_spans") or []) if s]
    if spans:
        return " ".join(spans)
    return None


def _parse_qa(question: str, qid: str, answers_obj: dict) -> QAExample:
    annotator_answers = answers_obj.get("answer", []) if answers_obj else []

    gold_answers: List[str] = []
    evidence: List[str] = []
    seen_ev = set()
    per_annotator_evidence: List[List[str]] = []
    unanswerable_flags: List[bool] = []

    for ann in annotator_answers:
        unanswerable_flags.append(bool(ann.get("unanswerable")))

        ans_str = _answer_string(ann)
        if ans_str and ans_str not in gold_answers:
            gold_answers.append(ans_str)

        ev = [e for e in (ann.get("evidence") or []) if e]
        per_annotator_evidence.append(ev)
        for e in ev:
            if e not in seen_ev:
                seen_ev.add(e)
                evidence.append(e)

    is_unanswerable = bool(annotator_answers) and all(unanswerable_flags)
    if is_unanswerable:
        gold_answers = []
        evidence = []

    return QAExample(
        question_id=qid,
        question=question,
        gold_answers=gold_answers,
        evidence=evidence,
        is_unanswerable=is_unanswerable,
        extra={"gold_evidence_per_annotator": per_annotator_evidence},
    )


def parse_qasper_row(row: dict) -> Document:
    """Parse one ``allenai/qasper`` row into a ``Document``."""
    qas = row.get("qas", {}) or {}
    questions = qas.get("question", []) or []
    qids = qas.get("question_id", []) or []
    answers = qas.get("answers", []) or []

    examples: List[QAExample] = []
    for i in range(len(questions)):
        qid = qids[i] if i < len(qids) else f"q{i}"
        ans_obj = answers[i] if i < len(answers) else {}
        examples.append(_parse_qa(questions[i], qid, ans_obj))

    return Document(
        doc_id=str(row.get("id", "")),
        title=row.get("title", "") or "",
        text=_build_text(row),
        questions=examples,
    )


class QASPERLoader(DatasetLoader):
    name = "qasper"

    def __init__(self, split: str = "test"):
        self.split = split

    def load(
        self,
        subset_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Document]:
        from datasets import load_dataset

        ds = load_dataset("allenai/qasper", split=self.split)
        docs = [parse_qasper_row(row) for row in ds]

        if subset_ids is not None:
            wanted = set(subset_ids)
            docs = [d for d in docs if d.doc_id in wanted]
        if limit is not None:
            docs = docs[:limit]
        return docs
