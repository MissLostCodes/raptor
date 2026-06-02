"""NarrativeQA generation metrics with the RAPTOR paper's AllenNLP tweaks.

Reference: Sarthi et al. (2024), RAPTOR (arXiv:2401.18059v1), which follows
the NarrativeQA (Kocisky et al., 2018) evaluation and applies three
modifications to the AllenNLP metric implementations:

  MOD 1: BLEU uses SmoothingFunction().method1 so that zero higher-order
         n-gram overlap does not hard-zero the score.
  MOD 2: BLEU-1 weights=(1,0,0,0) and BLEU-4 weights=(0.25,)*4.
  MOD 3: METEOR is computed over *tokenized* prediction and references.

ROUGE-L uses the ``rouge_score`` library (rougeL, use_stemmer=True), F-measure,
taking the max over references.
"""

from typing import List, Optional

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

_ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
_SMOOTH = SmoothingFunction().method1

_METEOR_READY = False


def _tokenize(text: str) -> List[str]:
    """Simple lowercase word tokenizer (no external data dependency)."""
    return text.lower().split()


def rouge_l(prediction: str, references: List[str]) -> float:
    """Max ROUGE-L F-measure over references, range 0..1."""
    if not references:
        return 0.0
    best = 0.0
    for ref in references:
        score = _ROUGE_SCORER.score(ref, prediction)["rougeL"].fmeasure
        if score > best:
            best = score
    return best


def bleu_1(prediction: str, references: List[str]) -> float:
    """Smoothed BLEU-1 (weights=(1,0,0,0)), max over references handled by NLTK."""
    hyp = _tokenize(prediction)
    refs = [_tokenize(r) for r in references]
    if not hyp or not refs:
        return 0.0
    return float(
        sentence_bleu(refs, hyp, weights=(1, 0, 0, 0), smoothing_function=_SMOOTH)
    )


def bleu_4(prediction: str, references: List[str]) -> float:
    """Smoothed BLEU-4 (weights=(0.25,)*4), max over references handled by NLTK."""
    hyp = _tokenize(prediction)
    refs = [_tokenize(r) for r in references]
    if not hyp or not refs:
        return 0.0
    return float(
        sentence_bleu(
            refs,
            hyp,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=_SMOOTH,
        )
    )


def _ensure_meteor_data() -> None:
    """Lazily make sure NLTK METEOR data is present.

    Raises RuntimeError (which tests treat as skip) if it cannot be obtained.
    """
    global _METEOR_READY
    if _METEOR_READY:
        return

    import nltk

    required = [
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("tokenizers/punkt", "punkt"),
    ]
    for find_path, pkg in required:
        try:
            nltk.data.find(find_path)
        except LookupError:
            try:
                ok = nltk.download(pkg, quiet=True)
            except Exception as exc:  # network / IO failure
                raise RuntimeError(
                    f"METEOR data '{pkg}' unavailable (download failed): {exc}"
                ) from exc
            if not ok:
                raise RuntimeError(
                    f"METEOR data '{pkg}' unavailable (download returned False)."
                )

    # Final sanity check that wordnet is actually importable.
    try:
        from nltk.corpus import wordnet  # noqa: F401

        wordnet.ensure_loaded()
    except Exception as exc:
        raise RuntimeError(f"METEOR data could not be loaded: {exc}") from exc

    _METEOR_READY = True


def meteor(prediction: str, references: List[str]) -> float:
    """METEOR over tokenized prediction/references.

    Raises RuntimeError if NLTK METEOR data is unavailable offline.
    """
    _ensure_meteor_data()
    from nltk.translate.meteor_score import meteor_score

    hyp = _tokenize(prediction)
    refs = [_tokenize(r) for r in references]
    if not hyp or not refs:
        return 0.0
    return float(meteor_score(refs, hyp))


def narrativeqa_metrics(prediction: str, references: List[str]) -> dict:
    """Return {rouge_l, bleu_1, bleu_4, meteor}; meteor=None if unavailable."""
    try:
        meteor_val: Optional[float] = meteor(prediction, references)
    except RuntimeError:
        meteor_val = None
    return {
        "rouge_l": rouge_l(prediction, references),
        "bleu_1": bleu_1(prediction, references),
        "bleu_4": bleu_4(prediction, references),
        "meteor": meteor_val,
    }
