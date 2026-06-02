"""Deterministic surface-feature structure score for routing in AHC.

`structure_score(text)` inspects only cheap, regex-based surface features of the raw
text (no LLM, no embeddings, no randomness) and returns a scalar in [0, 1] estimating
how strongly the document exposes an explicit heading structure, plus the feature dict
used to compute it.

Score formula (documented):

    heading_line_frac_scaled = min(1.0, heading_line_frac / 0.15)
    score = clip(
        0.70 * heading_line_frac_scaled
      + 0.15 * (1.0 if has_toc else 0.0)
      + 0.15 * heading_spacing_regularity,
        0.0, 1.0,
    )

`heading_line_frac` (the fraction of non-empty lines that look like a heading) is the
dominant signal: it saturates at a 15% heading density, so a doc with >= ~15% heading
lines already maxes that term. A table of contents and regular heading spacing each add
a smaller structural bonus. Empty / whitespace-only input scores 0.0.
"""

import re
from statistics import mean, pstdev
from typing import Dict, List, Tuple

_NUMBERED_RE = re.compile(r"^\s*\d+(\.\d+)*\.?\s+\S")
_MARKDOWN_RE = re.compile(r"^\s*#{1,6}\s+\S")
_KEYWORD_RE = re.compile(r"^\s*(chapter|section|appendix|part)\b", re.IGNORECASE)
_LETTER_RE = re.compile(r"[A-Za-z]")
_DOTTED_LEADER_RE = re.compile(r"\.{3,}\s*\d+\s*$")


def _is_allcaps_short(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return False
    if not _LETTER_RE.search(stripped):
        return False
    return stripped == stripped.upper()


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def structure_score(text: str) -> Tuple[float, Dict]:
    """Return (score in [0,1], features dict). Deterministic; no LLM/randomness."""
    features: Dict = {
        "numbered_heading_frac": 0.0,
        "markdown_heading_frac": 0.0,
        "allcaps_short_frac": 0.0,
        "keyword_heading_frac": 0.0,
        "heading_line_frac": 0.0,
        "has_toc": False,
        "heading_spacing_regularity": 0.0,
    }

    if not text or not text.strip():
        return 0.0, features

    lines = text.splitlines()
    # index non-empty lines (keep original line number for spacing regularity)
    nonempty: List[Tuple[int, str]] = [
        (i, ln) for i, ln in enumerate(lines) if ln.strip()
    ]
    n = len(nonempty)
    if n == 0:
        return 0.0, features

    numbered = 0
    markdown = 0
    allcaps = 0
    keyword = 0
    heading_line_numbers: List[int] = []

    for line_no, ln in nonempty:
        is_numbered = bool(_NUMBERED_RE.match(ln))
        is_markdown = bool(_MARKDOWN_RE.match(ln))
        is_allcaps = _is_allcaps_short(ln)
        is_keyword = bool(_KEYWORD_RE.match(ln))
        if is_numbered:
            numbered += 1
        if is_markdown:
            markdown += 1
        if is_allcaps:
            allcaps += 1
        if is_keyword:
            keyword += 1
        if is_numbered or is_markdown or is_allcaps or is_keyword:
            heading_line_numbers.append(line_no)

    heading_count = len(heading_line_numbers)

    features["numbered_heading_frac"] = numbered / n
    features["markdown_heading_frac"] = markdown / n
    features["allcaps_short_frac"] = allcaps / n
    features["keyword_heading_frac"] = keyword / n
    features["heading_line_frac"] = heading_count / n

    # has_toc heuristic: a "table of contents"/"contents" line near the top, OR several
    # dotted-leader lines like "Introduction ........ 12".
    top_window = nonempty[: min(15, n)]
    toc_phrase = any(
        "table of contents" in ln.lower() or ln.strip().lower() == "contents"
        for _, ln in top_window
    )
    dotted_leaders = sum(1 for _, ln in nonempty if _DOTTED_LEADER_RE.search(ln))
    features["has_toc"] = bool(toc_phrase or dotted_leaders >= 3)

    # heading_spacing_regularity: higher when gaps between heading line-numbers are
    # regular (low coefficient of variation). 0 if fewer than 2 headings.
    if heading_count >= 2:
        gaps = [
            heading_line_numbers[i + 1] - heading_line_numbers[i]
            for i in range(heading_count - 1)
        ]
        mu = mean(gaps)
        if mu > 0:
            cv = pstdev(gaps) / mu
            features["heading_spacing_regularity"] = _clip(1.0 - cv, 0.0, 1.0)
        else:
            features["heading_spacing_regularity"] = 0.0

    heading_line_frac_scaled = min(1.0, features["heading_line_frac"] / 0.15)
    score = (
        0.70 * heading_line_frac_scaled
        + 0.15 * (1.0 if features["has_toc"] else 0.0)
        + 0.15 * features["heading_spacing_regularity"]
    )
    return _clip(score, 0.0, 1.0), features
