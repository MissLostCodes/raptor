# Result — Second-corpus generality: the negative result replicates on QuALITY (n=30)

**Date:** 2026-06-30 · **Branch:** `ckraptor` · **Dataset:** QuALITY (validation), 30 articles,
448 scored questions · **Compute:** SBERT-only, no LLM · **Notebook:**
`notebooks/second_corpus_calibration.ipynb` · **Summary JSON:**
`raptor_runs/quality_second_corpus_summary.json`

Executes Phase 6 of `docs/specs/2026-06-27-weighted-adaptive-chunking-plan.md`: does the QASPER
refutation (cheap document features don't predict per-document optimal leaf size) generalize to a
second, very different corpus? **Yes** — on narrative multiple-choice text it replicates, with an
informative nuance (the features are somewhat stronger here but the headroom is smaller, so it lands
at near-parity rather than a clear loss).

## 0. Proxy note (read first)

QASPER ships gold-evidence paragraphs; QuALITY does not. We therefore use an **answer-recall
proxy** (`granularity_sweep.answer_coverage`): per question, the best token-recall of the gold
answer (the correct option's text) within the retrieved context; per-doc optimal size is the size
maximizing mean answer-recall. This is a **weaker proxy** than gold-evidence coverage (options are
often paraphrases), so we check it is non-degenerate before reading the gate:

- mean answer-recall across (doc, size) = **0.620** (healthy — not pinned at 0 or 1).
- median document length = **4685 tokens** (long enough that leaf size is a real choice).

## 1. H1 — headroom exists, but smaller than QASPER

| | QuALITY (n=30) | QASPER (n=50, ref.) |
|---|---:|---:|
| best fixed size | 400 tok | 200 tok |
| best fixed coverage | 0.6227 | 0.7362 |
| per-doc oracle | 0.6414 | 0.8080 |
| **relative headroom** | **+3.0%** | +9.8% |

Per-doc optimal-size distribution: `{50:9, 100:7, 150:2, 200:3, 300:3, 400:6}` — still spread
across the grid (heterogeneous), but the oracle gain over a tuned fixed size is only a third of
QASPER's. There is less to win here.

## 2. The negative result replicates

Per-feature Spearman vs per-document optimal size (top rows):

| feature | ρ vs optimal size |
|---|---:|
| `nonstopword_ratio` (surface) | +0.396 |
| `mean_segment_len_tokens` (A4, locality) | −0.396 |
| `topic_shift_rate` (A3, locality) | +0.379 |
| `mean_word_len_chars_norm` (surface) | −0.315 |
| `mean_sentence_len_tokens_norm` (surface) | −0.297 |
| `type_token_ratio` (surface) | −0.212 |

Best single feature |ρ| = **0.40**, still below the $|\rho|\gtrsim0.5$ bar the power analysis set.
Held-out over 50 random 70/30 splits:

| policy | mean TEST coverage | std |
|---|---:|---:|
| best global fixed (400 tok) | **0.6268** | 0.021 |
| selected feature → size | 0.6260 | 0.023 |
| per-doc oracle (ceiling) | 0.6457 | 0.022 |

- Selected feature beats fixed in **26/50 splits (52%)** — a coin flip — and mean coverage is
  within 0.001 of fixed. So on QuALITY the cheap-feature policy is **non-inferior but not better**:
  it does not beat a tuned global size.
- Pre-registered gate: no feature reaches |ρ| ≥ 0.5, and the selected map does not beat fixed →
  **negative result replicates.**

## 3. Cross-corpus nuance (a real finding, not noise)

The semantic-locality features carry **more** signal on narrative prose than on scientific papers:
`mean_segment_len_tokens` (A4) is −0.396 here vs −0.067 on QASPER, and `topic_shift_rate` is +0.379
vs +0.089. This is consistent with the locality hypothesis being *more* applicable to narrative
text (where semantic segments are a more meaningful unit) — yet the correlation still tops out near
0.40, short of a reliable win, and the small headroom (+3%) means even the oracle barely beats
fixed. So the qualitative conclusion is unchanged across two very different corpora.

## 4. Takeaway

> Across QASPER (scientific, evidence QA) and QuALITY (narrative, multiple-choice), cheap
> document-intrinsic features — surface density and semantic locality alike — do **not** beat a
> single tuned global leaf size. QASPER has large headroom the features cannot see (a clear loss);
> QuALITY has small headroom and the features reach parity but no win. The practical advice
> generalizes: **tune one global leaf size** (≈200 tok on QASPER, ≈400 tok on QuALITY).

**Honest limits:** the QuALITY result rests on the weaker answer-recall proxy and a near-parity
margin; it is a *non-inferiority* generalization, not a second decisive refutation. It nonetheless
addresses the single-corpus threat to validity: the refutation is not QASPER-specific.

## 5. Pointers
- Plan: `docs/specs/2026-06-27-weighted-adaptive-chunking-plan.md`
- QASPER locality result: `docs/results/2026-06-27-locality-calibration.md`
- Proxy + sweep: `experiments/granularity_sweep.py` (`answer_coverage`); notebook:
  `notebooks/second_corpus_calibration.ipynb`
