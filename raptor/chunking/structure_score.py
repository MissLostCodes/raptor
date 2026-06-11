"""Deterministic surface-feature structure score for routing in AHC.

`structure_score(text)` inspects only cheap, regex-based surface features of the raw
text (no LLM, no embeddings, no randomness) and returns a scalar in [0, 1] estimating
how strongly the document exposes an explicit heading structure, plus the feature dict
used to compute it.

Score formula (documented):

    heading_line_frac_scaled = min(
        1.0, max(heading_line_frac, generic_heading_frac) / 0.15
    )
    score = clip(
        0.70 * heading_line_frac_scaled
      + 0.15 * (1.0 if has_toc else 0.0)
      + 0.15 * heading_spacing_regularity,
        0.0, 1.0,
    )

`heading_line_frac` (the fraction of non-empty lines that look like a numbered,
markdown, short ALL-CAPS, or keyword heading) is the dominant signal: it saturates at a
15% heading density, so a doc with >= ~15% heading lines already maxes that term.
`generic_heading_frac` is a complementary detector for plain Title Case section headers
(e.g. scientific papers: ``Introduction``, ``Related Work``) that carry no numbering or
markdown markers; the dominant term takes the max of the two fractions. A table of
contents and regular heading spacing each add a smaller structural bonus. Empty /
whitespace-only input scores 0.0.
"""

import re
from statistics import mean, pstdev
from typing import Dict, List, Tuple

_NUMBERED_RE = re.compile(r"^\s*\d+(\.\d+)*\.?\s+\S")
_MARKDOWN_RE = re.compile(r"^\s*#{1,6}\s+\S")
_KEYWORD_RE = re.compile(r"^\s*(chapter|section|appendix|part)\b", re.IGNORECASE)
_LETTER_RE = re.compile(r"[A-Za-z]")
_DOTTED_LEADER_RE = re.compile(r"\.{3,}\s*\d+\s*$")

# Generic Title Case heading detection.
_TERMINAL_PUNCT_RE = re.compile(r"[.?!,;]$")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
_GENERIC_STOPWORDS = {
    "of", "the", "and", "in", "for", "to", "a", "an", "with",
    "on", "by", "at", "from", "as", "or", "is", "are",
}


def _is_allcaps_short(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return False
    if not _LETTER_RE.search(stripped):
        return False
    return stripped == stripped.upper()


def _is_title_ish(stripped: str) -> bool:
    """True if a short line reads like a plain Title Case heading.

    Independent of surrounding lines: checks length, word count, terminal
    punctuation, minimum alphabetic content, and that >= 60% of the non-stopword
    "content" words start with an uppercase letter.
    """
    if len(stripped) > 64:
        return False
    if len(stripped.split()) > 12:
        return False
    if _TERMINAL_PUNCT_RE.search(stripped):
        return False
    if len(_LETTER_RE.findall(stripped)) < 2:
        return False
    words = _WORD_RE.findall(stripped)
    content = [w for w in words if w.lower() not in _GENERIC_STOPWORDS]
    if not content:
        return False
    upper = sum(1 for w in content if w[0].isupper())
    return upper / len(content) >= 0.6


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def structure_score(text: str) -> Tuple[float, Dict]:
    """Return (score in [0,1], features dict). Deterministic; no LLM/randomness."""
    features: Dict = {
        "numbered_heading_frac": 0.0,
        "markdown_heading_frac": 0.0,
        "allcaps_short_frac": 0.0,
        "keyword_heading_frac": 0.0,
        "generic_heading_frac": 0.0,
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

    # generic Title Case heading lines: title-ish, preceded by a blank line, and
    # followed by a strictly longer body line. Uses the full `lines` (blank lines
    # carry the context); fraction is over non-empty lines like the others.
    generic = 0
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            continue
        if not _is_title_ish(s):
            continue
        # preceded by a blank line (or start of document)
        if i != 0 and lines[i - 1].strip():
            continue
        # followed by a longer body (next non-empty line, stripped, longer than s)
        body_len = 0
        for j in range(i + 1, len(lines)):
            nxt = lines[j].strip()
            if nxt:
                body_len = len(nxt)
                break
        if body_len <= len(s):
            continue
        generic += 1

    features["numbered_heading_frac"] = numbered / n
    features["markdown_heading_frac"] = markdown / n
    features["allcaps_short_frac"] = allcaps / n
    features["keyword_heading_frac"] = keyword / n
    features["generic_heading_frac"] = generic / n
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

    heading_line_frac_scaled = min(
        1.0,
        max(features["heading_line_frac"], features["generic_heading_frac"]) / 0.15,
    )
    score = (
        0.70 * heading_line_frac_scaled
        + 0.15 * (1.0 if features["has_toc"] else 0.0)
        + 0.15 * features["heading_spacing_regularity"]
    )
    return _clip(score, 0.0, 1.0), features
