# H2 Preview — density features vs optimal leaf size (8-doc QASPER pilot)

**Date:** 2026-06-11 · **Branch:** `ckraptor` · **Status:** preview (n=8, directional only)
**Inputs:** the 8-doc granularity sweep (`notebooks/granularity_sweep.ipynb`, run on
Colab) + deterministic `density_score` features computed locally on the same 8 QASPER
documents. No SBERT/LLM in this analysis — it reuses the sweep's per-doc optimal sizes.

## TL;DR

The equal-weight composite density score **`rho` does not predict optimal leaf size**
(Spearman **+0.04**), but that is an **aggregation artifact, not absence of signal**:
individual density features are strongly predictive and point in **opposite directions**,
so the uniform mean cancels them. Selecting the single strongest feature on a training
split recovers the signal. This changes the calibration method — the headline
training-free arm now selects a feature rather than trusting the hand-weighted composite.

## The numbers (n=8)

Per-document optimal size (from the sweep): spans 50→400 tokens (H1, already confirmed).
Spearman rank-correlation of each deterministic density signal vs that optimal size:

| signal | Spearman vs optimal size |
|---|---:|
| `mean_word_len_chars_norm` | **+0.786** |
| `nonstopword_ratio` | **+0.663** |
| `type_token_ratio` | **−0.602** |
| `numeral_symbol_density` | −0.258 |
| `mean_sentence_len_tokens_norm` | +0.025 |
| `list_marker_density` | 0.000 |
| **`rho` (equal-weight composite)** | **+0.037** |

## Interpretation

1. **The composite cancels signal.** `rho` averages a **+0.79** feature
   (`mean_word_len`) with a **−0.60** feature (`type_token_ratio`); the opposite signs
   net out to ~0. The equal-weight aggregation in `density_score` (a documented
   placeholder) is therefore the wrong driver for sizing — the features carry the
   information, the mean destroys it.
2. **The `invert=True` "denser ⇒ smaller" assumption is only half-right.** Two
   "harder-text" signals — longer words and a higher content-word ratio — correlate with
   **larger** optimal leaves, the opposite direction. Lexical diversity (`type_token_ratio`)
   is the one signal matching "denser ⇒ smaller." Direction must be **calibrated per
   feature**, not assumed globally.
3. **`list_marker_density` and `mean_sentence_len` are dead** on this corpus (scientific
   papers have no lists; sentence length doesn't track granularity here). Expected; they
   may matter on other document types.

## Method change this motivates

`experiments/granularity_calibration.py` gains two functions:

- `feature_target_correlations(feat_by_doc, target_by_doc)` — the per-feature Spearman
  ranking above (the diagnostic).
- `select_and_fit(feat_by_doc, target_by_doc, train_ids)` — picks the top-|Spearman|
  feature **on the training split only** and fits the parsimonious `size = a + b·feature`
  (still closed-form, 1 feature + 2 params, no test leakage).

The calibration notebook's held-out table now reports **three** training-free-style rows —
best-global-fixed, composite-`rho`→size (shown to fail), and **selected-feature→size**
(the real arm) — against the learned 6-feature regressor and the per-doc oracle ceiling.

In-sample sanity check on the 8 pilot docs: `select_and_fit` picks
`mean_word_len_chars_norm` (+0.786) and its predicted sizes track the true optimum's
rank ordering far better than `rho` (which is flat).

## Honesty / limits

- **n=8 is a preview, not a result.** For n=8, |Spearman| ≈ 0.74 is roughly the p<0.05
  threshold, so only `mean_word_len` is borderline-significant; the rest are suggestive.
- **Feature selection on 6 candidates over 8 docs is mildly optimistic.** The held-out
  TEST split mitigates it, but the conclusion must be re-checked at the planned ~50-doc
  scale before any "ship this feature" claim. The fitted slope is also unstable at n=8
  (the normalized word-length feature has a narrow range), which large N will damp.
- This does **not** touch H1 (already passed) — it sharpens **H2** (cheap features predict
  optimal size) and tells us *which* features and *which directions* before the scaled run.

## Next

Run the scaled (~50-doc) sweep + `granularity_calibration.ipynb`. The H2 gate is: does
the **selected-feature** training-free map beat best-global-fixed and approach both the
learned regressor and the oracle on the **held-out** docs?
