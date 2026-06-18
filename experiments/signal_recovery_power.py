"""Signal-recovery power analysis for the density->size calibration (H2).

Why this exists
---------------
H2 fails on QASPER: the closed-form ``size = a + b * feature`` map (and a learned
6-feature regressor) lose to a single tuned global leaf size. Two explanations are
observationally distinct and matter for the paper:

  (A) the cheap density features carry no granularity signal, OR
  (B) the *calibration target* is too noisy to exploit a signal that is there --
      QASPER has only ~3.4 answerable questions per document, so each document's
      "optimal size" is an argmax over very few questions and is therefore a
      high-variance label.

This module is a **positive control + power analysis** that fully isolates (B). It
*plants* a known TTR->size relationship (so a signal provably exists), models the
finite-questions-per-doc noise explicitly, and measures how well the SAME
calibration recipe used in the notebook recovers it as a function of the number of
questions per document. Everything is offline (no SBERT, no LLM, no Drive): it
reuses only the pure-stdlib analytics in ``experiments.granularity_calibration``
and ``experiments.granularity_sweep``.

Headline use: calibrate the per-question noise so that at QASPER's questions/doc
the recovered Spearman matches the observed -0.38, then read off the win-rate the
method can achieve at that operating point, and how many questions/doc it would
take to turn the planted signal into a reliable win.
"""

from __future__ import annotations

import random
from statistics import mean, pstdev
from typing import Dict, List, Tuple

from experiments import granularity_calibration as gc
from experiments import granularity_sweep as gs

SWEPT_SIZES: List[int] = [50, 100, 150, 200, 300, 400]
FEATURE_NAMES = [
    "mean_sentence_len_tokens_norm",
    "type_token_ratio",
    "numeral_symbol_density",
    "mean_word_len_chars_norm",
    "list_marker_density",
    "nonstopword_ratio",
]

# Planted signal: optimal size as a function of type-token ratio. The slope/intercept
# are exactly the n=8-preview-style line the notebook's select_and_fit produced on
# QASPER (size ~ 373 - 926 * TTR), so the planted signal is of *realistic* strength
# for this corpus, not an exaggerated one.
SIGNAL_A = 373.0
SIGNAL_B = -926.0
TTR_LO, TTR_HI = 0.17, 0.36         # observed QASPER TTR range (from the real run)
PEAK_COV = 0.81                     # coverage at a document's true optimum
KAPPA = 0.07 / 150.0               # coverage lost per token away from the optimum


def _clip(v: float, lo: int = 50, hi: int = 400) -> int:
    return int(max(lo, min(hi, round(v))))


def _nearest_swept(size: float) -> int:
    return min(SWEPT_SIZES, key=lambda s: abs(s - size))


def synth_dataset(
    n_docs: int, questions_per_doc: int, sigma_q: float, rng: random.Random
) -> Tuple[List[dict], Dict[str, Dict[str, float]], Dict[str, int]]:
    """Build (records, feats, opt) with a planted TTR->size signal and finite-Q noise.

    Each document gets a latent optimum on the swept grid from the planted line. Its
    true tent-shaped coverage curve is sampled with ``questions_per_doc`` noisy
    per-question observations (per-question std ``sigma_q``); the document-level
    coverage is their mean, so the label noise shrinks like ``sigma_q / sqrt(Q)`` --
    exactly the mechanism by which few questions/doc blur the argmax.
    """
    records: List[dict] = []
    feats: Dict[str, Dict[str, float]] = {}
    opt: Dict[str, int] = {}
    for i in range(n_docs):
        doc = f"d{i}"
        ttr = rng.uniform(TTR_LO, TTR_HI)
        mu = _nearest_swept(_clip(SIGNAL_A + SIGNAL_B * ttr))  # true latent optimum
        # Dummy features carry no signal; only TTR is informative (the positive control).
        feats[doc] = {f: rng.random() for f in FEATURE_NAMES}
        feats[doc]["type_token_ratio"] = ttr
        for s in SWEPT_SIZES:
            true_cov = PEAK_COV - KAPPA * abs(s - mu)
            obs = mean(true_cov + rng.gauss(0.0, sigma_q) for _ in range(questions_per_doc))
            obs = max(0.0, min(1.0, obs))
            records.append(
                {"doc_id": doc, "size": s, "n_chunks": 10,
                 "n_questions": questions_per_doc, "mean_evidence_coverage": obs}
            )
    opt = gs.best_size_per_doc(records)  # the noisy argmax label the method must learn
    return records, feats, opt


def evaluate(
    records: List[dict], feats: Dict[str, Dict[str, float]], opt: Dict[str, int],
    n_splits: int = 50,
) -> Tuple[float, float, int, int]:
    """Win-rate and mean coverage of the continuous selected-feature map vs best fixed.

    Mirrors the notebook protocol exactly: fit on TRAIN, score on TEST against the
    same (noisy) records, repeat over ``n_splits`` random 70/30 splits.
    """
    bf, _ = gc.best_global_fixed(records)
    covs_map: List[float] = []
    wins = 0
    n = 0
    for seed in range(n_splits):
        tr, te = gc.train_test_split_docs(list(opt), seed=seed)
        if not tr or not te:
            continue
        n += 1
        sel = gc.select_and_fit(feats, opt, tr, feature_names=FEATURE_NAMES)
        if sel["feature"]:
            pred = {k: gc.predict_size(feats[k][sel["feature"]], sel["fit"], 50, 400) for k in te}
        else:
            pred = {k: bf for k in te}
        c = gc.coverage_under_sizes(records, pred)
        cf = gc.coverage_under_sizes(records, {k: bf for k in te})
        covs_map.append(c)
        wins += int(c > cf)
    return (mean(covs_map) if covs_map else float("nan"),
            pstdev(covs_map) if covs_map else 0.0, wins, n)


def recovered_spearman(feats: Dict[str, Dict[str, float]], opt: Dict[str, int]) -> float:
    """Spearman(TTR, estimated optimal size) -- the analogue of QASPER's observed -0.38."""
    ids = list(opt)
    return gc.rank_corr([feats[k]["type_token_ratio"] for k in ids], [opt[k] for k in ids])


def sweep_questions(
    q_grid: List[int], n_docs: int = 50, sigma_q: float = 0.25,
    n_datasets: int = 30, n_splits: int = 50, base_seed: int = 0,
) -> List[dict]:
    """For each questions-per-doc value, Monte-Carlo the recovered |Spearman| and win-rate."""
    rows = []
    for q in q_grid:
        rhos, winrates, means = [], [], []
        for m in range(n_datasets):
            rng = random.Random(base_seed + 1000 * q + m)
            records, feats, opt = synth_dataset(n_docs, q, sigma_q, rng)
            rhos.append(recovered_spearman(feats, opt))
            mu, _sd, wins, n = evaluate(records, feats, opt, n_splits)
            winrates.append(wins / n if n else float("nan"))
            means.append(mu)
        rows.append({
            "questions_per_doc": q,
            "recovered_spearman": mean(rhos),
            "winrate": mean(winrates),
            "winrate_std": pstdev(winrates),
            "mean_cov": mean(means),
        })
    return rows


def main() -> None:
    sigma_q = 0.25
    q_grid = [3, 4, 5, 8, 12, 20, 40, 100]
    print(f"Signal-recovery power analysis (planted TTR->size, per-question sigma={sigma_q})")
    print("Positive control: a real signal IS present; can the method recover it as")
    print("questions-per-doc grows? QASPER operates at ~3.4 questions/doc, rho~=-0.38.\n")
    print(f"{'Q/doc':>6}{'recovered rho':>16}{'win-rate vs fixed':>20}{'mean cov':>11}")
    print("-" * 53)
    for r in sweep_questions(q_grid, sigma_q=sigma_q):
        print(f"{r['questions_per_doc']:>6}{r['recovered_spearman']:>16.3f}"
              f"{r['winrate']:>18.0%}  {r['mean_cov']:>9.4f}")
    print("\nRe/ QASPER: find the Q row whose recovered rho ~ -0.38; its win-rate is what the")
    print("method can achieve at QASPER's noise level. Higher Q rows show how much more")
    print("evidence-per-document it would take to convert the planted signal into a win.")


if __name__ == "__main__":
    main()
