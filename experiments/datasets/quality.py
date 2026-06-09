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


# Field-name variants seen across QuALITY HF mirrors. Canonical names first.
# Hardening these prevents a silent metric failure: if the gold field is not
# found, every ``gold_index`` is None and accuracy reads a silent 0; if the
# hard field is not found, the QuALITY-HARD subset is silently empty.
_GOLD_KEYS = ("gold_label", "gold", "answer", "label")
_HARD_KEYS = ("difficult", "hard", "is_hard")
_ARTICLE_KEYS = ("article", "context", "passage", "text")


def _article_id(obj: dict) -> str:
    return str(obj.get("article_id") or obj.get("set_unique_id") or "")


def _article_text(obj: dict) -> str:
    for key in _ARTICLE_KEYS:
        v = obj.get(key)
        if v:
            return v
    return ""


def _gold_index(q: dict):
    """Return the 0-based gold option index, or ``None`` if absent.

    The QuALITY gold answer is **1-based** (its convention across mirrors), so a
    stored value ``v`` maps to index ``v - 1``. Booleans are ignored (a ``hard``
    flag must not be mistaken for a gold label). Returns ``None`` on the test
    split, where gold labels are withheld.
    """
    for key in _GOLD_KEYS:
        val = q.get(key)
        if isinstance(val, bool):
            continue
        if isinstance(val, int):
            return val - 1
    return None


def _is_hard(q: dict) -> bool:
    for key in _HARD_KEYS:
        if key in q and q[key] is not None:
            return bool(q[key])
    return False


def _parse_question(q: dict) -> QAExample:
    return QAExample(
        question_id=str(q.get("question_unique_id") or q.get("question", "")),
        question=q.get("question", "") or "",
        options=list(q.get("options", []) or []),
        gold_index=_gold_index(q),
        is_hard=_is_hard(q),
    )


def parse_quality_article(obj: dict) -> Document:
    """Parse one article-level QuALITY object into a ``Document``."""
    questions = [_parse_question(q) for q in (obj.get("questions", []) or [])]
    return Document(
        doc_id=_article_id(obj),
        title=obj.get("title", "") or "",
        text=_article_text(obj),
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
                    text=_article_text(row),
                    questions=[],
                )
                order.append(aid)
            grouped[aid].questions.append(_parse_question(row))

        return article_level + [grouped[a] for a in order]
