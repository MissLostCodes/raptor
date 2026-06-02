"""NarrativeQA loader (full-story abstractive QA; ROUGE-L / BLEU / METEOR).

Assumed HuggingFace schema (``deepmind/narrativeqa``), one row per question::

    {
      "document": {
          "id":      str,
          "summary": {"text": str, ...},
          "text":    str,                   # FULL story (html/plain text)
      },
      "question": {"text": str, ...},
      "answers":  [ {"text": str, ...}, {"text": str, ...} ],  # 2 references
    }

``group_narrativeqa_rows`` groups rows by ``document.id`` into one
``Document`` (``text`` = the full story) holding every question, with each
question's two reference answers as ``gold_answers``.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from experiments.datasets.base import DatasetLoader, Document, QAExample


def group_narrativeqa_rows(rows: Iterable[dict]) -> List[Document]:
    """Group per-question NarrativeQA rows into per-story ``Document``s."""
    by_doc: dict = {}
    order: List[str] = []

    for idx, row in enumerate(rows):
        doc = row.get("document", {}) or {}
        doc_id = str(doc.get("id", ""))

        if doc_id not in by_doc:
            summary = doc.get("summary", {}) or {}
            by_doc[doc_id] = Document(
                doc_id=doc_id,
                title=doc.get("title", "") or "",
                text=doc.get("text", "") or "",
                questions=[],
                meta={"summary": summary.get("text", "") or ""},
            )
            order.append(doc_id)

        question = row.get("question", {}) or {}
        gold = [a.get("text", "") for a in (row.get("answers", []) or [])]

        by_doc[doc_id].questions.append(
            QAExample(
                question_id=f"{doc_id}_q{len(by_doc[doc_id].questions)}",
                question=question.get("text", "") or "",
                gold_answers=gold,
            )
        )

    return [by_doc[d] for d in order]


class NarrativeQALoader(DatasetLoader):
    name = "narrativeqa"

    def __init__(self, split: str = "test"):
        self.split = split

    def load(
        self,
        subset_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Document]:
        from experiments.datasets.base import load_hf_split

        ds = load_hf_split("deepmind/narrativeqa", split=self.split)
        docs = group_narrativeqa_rows(ds)

        if subset_ids is not None:
            wanted = set(subset_ids)
            docs = [d for d in docs if d.doc_id in wanted]
        if limit is not None:
            docs = docs[:limit]
        return docs
