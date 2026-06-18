# Signal-Recovery Power Analysis — why the density→size formula fails (positive control)

**Date:** 2026-06-18 · **Branch:** `ckraptor` · **Status:** supporting analysis for the
H2 negative result. **Fully offline** (no SBERT, no LLM, no Drive) — reproduce with
`python -m experiments.signal_recovery_power`.

## Question

H2 showed the closed-form `size = a + b·feature` map (and a learned 6-feature regressor)
**lose to a tuned global fixed leaf size** on QASPER (selected-feature win-rate 7/50 = 14%,
mean 0.704 vs 0.729). The natural follow-up question — the one a guide will ask — is
**"can we write a formula that gives optimal chunk size from a document?"** The honest
answer is *we did, and it loses*. This analysis explains **why**, and rules out the most
obvious "it's just a bug / too little data" objections.

Two observationally distinct causes:
- **(A)** the cheap density features carry **no clean exploitable** granularity signal; or
- **(B)** the calibration **target is too noisy** — QASPER has only ~3.4 answerable
  questions/doc, so each doc's "optimal size" is an argmax over very few questions.

## Method (positive control + power sweep)

We **plant** a known signal `optimal_size = 373 − 926·TTR` (exactly the line the notebook's
`select_and_fit` produced on QASPER, so the planted signal is of *realistic strength*), give
each synthetic doc a tent-shaped coverage curve peaking at its planted optimum, and sample
it with `Q` noisy per-question observations (per-question std `σ_q`; doc-level label noise
therefore shrinks like `σ_q/√Q`). We then run the **same** calibration recipe used in the
notebook (`select_and_fit` → `predict_size`, 50 random 70/30 splits, scored on the same
noisy records) and measure (i) the **recovered Spearman**(TTR, estimated optimal size) — the
analogue of QASPER's observed −0.38 — and (ii) the **win-rate vs best fixed**. Monte-Carlo
averaged over 30 synthetic datasets per cell. Code: `experiments/signal_recovery_power.py`.

## Result 1 — the method is NOT broken (positive control passes)

With a planted signal present, the calibration recovers it and beats fixed **monotonically**
as questions/doc grows (`σ_q = 0.25`):

| questions/doc | recovered ρ | win-rate vs fixed | mean coverage |
|---:|---:|---:|---:|
| 3   | −0.222 | 34% | 0.7827 |
| 5   | −0.216 | 39% | 0.7877 |
| 8   | −0.309 | 46% | 0.7954 |
| 12  | −0.351 | 43% | 0.7939 |
| 20  | −0.447 | 62% | 0.7993 |
| 40  | −0.602 | 83% | 0.8039 |
| 100 | −0.707 | 95% | 0.8053 |

So the failure on QASPER is **not** a degenerate estimator: when a genuine TTR→size signal
exists, the closed-form map captures it (win-rate → 95%). This is the positive control the
negative result needs.

## Result 2 — QASPER underperforms even a *clean* signal of the same strength → cause (A)

The key invariant — **win-rate as a function of recovered |ρ|** — is stable across the
per-question noise `σ_q`. Interpolating each curve to QASPER's observed correlation
(|ρ| = 0.38):

| per-question noise `σ_q` | clean-signal win-rate at \|ρ\|=0.38 |
|---:|---:|
| 0.15 | 54% |
| 0.25 | 49% |
| 0.35 | 49% |
| **QASPER (actual)** | **14%** |

A **clean** linear TTR→size signal at QASPER's measured correlation would win **~half** the
splits. QASPER wins only **14%** — a ~35-point shortfall that is **robust to the noise
assumption**. Because the synthetic model is *perfectly specified* (the fitted form equals
the data-generating form), its win-rates are an **upper bound** on what is achievable at a
given correlation; QASPER falling far below that bound means its −0.38 correlation is **not a
clean, monotone, exploitable linear relationship** — it is noisier / non-linear /
outlier-driven than the raw number suggests.

This favors **cause (A)** over (B): the bottleneck is the **feature's relationship to optimal
size**, not merely label noise from few questions. Reaching a *reliable* win (≥80%) needs
|ρ| ≈ 0.6 (the `Q=40` row) — far stronger than **any** feature we have (best is TTR at 0.38).
More questions/doc alone would not close the gap.

## Takeaways for the paper

1. **The "formula" question is answered and closed.** A document-level closed-form leaf-size
   formula `size ≈ 373 − 926·TTR` exists but is **worse than a single tuned constant**
   (≈200 tok). The defensible recommendation is a tuned global size.
2. **The negative result is firm, not an artifact.** The method recovers planted signal
   (positive control), so QASPER's failure is about the **features/target**, not the
   estimator — and QASPER underperforms even a clean signal of equal correlation, pointing to
   genuinely weak/non-linear feature signal rather than just noisy labels.
3. **What it would take.** A *reliable* density→size win needs a feature correlating with
   optimal size at |ρ| ≳ 0.6. None of the six cheap surface features come close. Future work
   = a stronger (likely richer / query-aware / learned-multi-signal) predictor, not a cheaper
   one — consistent with Mix-of-Granularity / HiChunk being learned.

## Honesty / limits

- **Synthetic model.** Tent-shaped curves, uniform TTR over the observed range, Gaussian
  per-question noise. The **positive control** (Result 1) is assumption-light; the
  **quantitative QASPER comparison** (Result 2) is model-dependent, which is why we report it
  as *corroborating* evidence and verify the win-rate–vs–|ρ| invariant across `σ_q`.
- This analysis does not touch H1 (the oracle headroom is real and unchanged). It explains
  why *these features* cannot capture that headroom.
