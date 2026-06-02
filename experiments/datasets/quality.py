"""QuALITY loader (multiple-choice over articles; accuracy + HARD subset).

Assumed schema (article-level object)::

    {
      "article_id" | "set_unique_id": str,
      "title":   str,
      "article": str,                       # full article text
      "questions": [ {
          "question":  str,
          "options":   [str, str, str, str],
          "gold_label": int,                # 1-based; absent on the test split
          "difficult":  0 | 1,
      }, ... ],
    }

``parse_quality_article`` builds one ``QAExample`` per question with
``gold_index = gold_label - 1`` (or ``None`` when ``gold_label`` is missing,
e.g. the held-out test split) and ``is_hard = bool(difficult)``.

The HuggingFace mirror (default ``emozilla/quality``) may be flattened to
one row per question; ``QuALITYLoader.load`` therefore normalizes both
article-level and question-level rows, grouping question-level rows by their
article id before parsing.
"""

from __future__ import annotations

from typing import List, Optional

from experiments.datasets.base import DatasetLoader, Document, QAExample


def _article_id(obj: dict) -> str:
    return str(obj.get("article_id") or obj.get("set_unique_id") or "")


def _parse_question(q: dict) -> QAExample:
    gold_label = q.get("gold_label")
    gold_index = (gold_label - 1) if isinstance(gold_label, int) else None
    return QAExample(
        question_id=str(q.get("question_unique_id") or q.get("question", "")),
        question=q.get("question", "") or "",
        options=list(q.get("options", []) or []),
        gold_index=gold_index,
        is_hard=bool(q.get("difficult", 0)),
    )


def parse_quality_article(obj: dict) -> Document:
    """Parse one article-level QuALITY object into a ``Document``."""
    questions = [_parse_question(q) for q in (obj.get("questions", []) or [])]
    return Document(
        doc_id=_article_id(obj),
        title=obj.get("title", "") or "",
        text=obj.get("article", "") or "",
        questions=questions,
    )


class QuALITYLoader(DatasetLoader):
    name = "quality"

    def __init__(self, split: str = "validation", hf_name: str = "emozilla/quality"):
        self.split = split
        self.hf_name = hf_name

    def load(
        self,
        subset_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Document]:
        from experiments.datasets.base import load_hf_split

        ds = load_hf_split(self.hf_name, split=self.split)
        docs = self._normalize(ds)

        if subset_ids is not None:
            wanted = set(subset_ids)
            docs = [d for d in docs if d.doc_id in wanted]
        if limit is not None:
            docs = docs[:limit]
        return docs

    @staticmethod
    def _normalize(rows) -> List[Document]:
        """Handle either article-level rows or one-row-per-question mirrors."""
        article_level: List[Document] = []
        grouped: dict = {}
        order: List[str] = []

        for row in rows:
            row = dict(row)
            if isinstance(row.get("questions"), list):
                # Article-level row.
                article_level.append(parse_quality_article(row))
                continue

            # Question-level row: group by article id.
            aid = _article_id(row)
            if aid not in grouped:
                grouped[aid] = Document(
                    doc_id=aid,
                    title=row.get("title", "") or "",
                    text=row.get("article", "") or "",
                    questions=[],
                )
                order.append(aid)
            grouped[aid].questions.append(_parse_question(row))

        return article_level + [grouped[a] for a in order]
