# Result — Evidence-locality features do not predict optimal leaf size either (n=50)

**Date:** 2026-06-27 · **Branch:** `ckraptor` · **Dataset:** QASPER, 50 papers (same cached
sweep as H1/H2) · **Compute:** SBERT-only, no LLM · **Notebook:** `notebooks/locality_calibration.ipynb`
· **Summary JSON:** `raptor_runs/locality_calibration_summary.json`

Executes Phases 0 + 2 of `docs/specs/2026-06-27-weighted-adaptive-chunking-plan.md`. The H2
result (`docs/results/2026-06-15-h2-density-50doc.md`) refuted the cheap **surface** density
features. This run tested the **causally-motivated** *evidence-locality / semantic-segment*
features (A1–A5, B1, C4 in `raptor/chunking/segment_features.py`) plus the **label-derived target
diagnostics** (E1/E2/E3). **The cheap, document-intrinsic, training-free route is now decisively
dead** — and the diagnostics tell us *why*.

---

## 1. Headroom replicated (H1 still holds)

| | value |
|---|---:|
| best fixed size | **200 tok** |
| best fixed coverage | 0.7362 |
| per-doc oracle coverage | 0.8080 |
| absolute gap | 0.0718 |
| **relative headroom** | **+9.75%** |

Per-doc optimal-size distribution: `{50:17, 100:11, 150:6, 200:8, 300:3, 400:5}` — mass at every
size. The adaptive opportunity is real and unchanged.

## 2. The decisive new finding — E2 ≈ 0: the latent driver doesn't drive

The plan pre-registered: *"❌ Dead if even the label-derived E2/E3 diagnostics don't explain
optimal size."* They don't.

| diagnostic (label-derived, near-oracle) | Spearman vs per-doc optimal size |
|---|---:|
| **E2 — gold-evidence span length** | **+0.005** (≈ 0) |
| E1 — flat-curve fraction | 22% of docs are size-insensitive (flatness < 0.05) |

Gold-evidence span length — the quantity we hypothesized *is* the optimal leaf size — has **zero**
rank correlation with the measured optimal size. This is much stronger than "we couldn't find a
cheap feature": the **target itself does not respond to the mechanism we theorized**. No cheap
document feature can predict a target that even the near-oracle label-derived signal cannot.

## 3. The causally-right features are no better than surface (and A4 failed)

Per-feature Spearman vs per-doc optimal size (n=50):

| feature (tier) | ρ vs optimal size | ρ vs E2 (latent driver) |
|---|---:|---:|
| `gzip_ratio` (C4) | **−0.249** | +0.074 |
| `paragraph_len_cv` (B1) | +0.218 | −0.111 |
| `segment_len_cv` (A5) | +0.218 | −0.174 |
| `topic_shift_rate` (A3) | +0.089 | +0.036 |
| `mean_adjacent_cosine` (A1) | −0.080 | +0.059 |
| `var_adjacent_cosine` (A2) | −0.093 | −0.011 |
| **`mean_segment_len_tokens` (A4)** | **−0.067** | −0.028 |
| `paragraph_len_mean` (B1) | +0.016 | **+0.452** |

Two things stand out:

1. **A4 — the headline hypothesis (H-new) — is refuted.** Mean semantic-segment length, the
   "direct estimate of the natural chunk size," correlates with optimal size at **−0.067** (noise).
   Topic-shift rate and adjacent-cosine variance are equally dead.
2. **The causal chain breaks at E2→size, not feature→E2.** `paragraph_len_mean` *does* track the
   gold-evidence span length (**ρ = 0.452**) — features can see evidence geometry. But since E2
   itself doesn't predict optimal size, that link is wasted. The failure is **not** "features can't
   measure document structure"; it is that **document structure / evidence geometry does not
   determine the best leaf size** in this pipeline.

For reference, the strongest single feature found across *both* rounds remains the old surface
`type_token_ratio` at |ρ| = 0.379 — and the power analysis already showed that is too weak to win.
The causally-motivated features correlate **less**, not more.

## 4. Held-out 50-split gate — still loses to a tuned constant

| policy (mean TEST coverage, 50 random 70/30 splits) | mean | std |
|---|---:|---:|
| **best global fixed (200 tok)** | **0.7291** | 0.043 |
| NEW selected locality feature → size | 0.7098 | 0.050 |
| per-doc oracle (ceiling) | 0.8061 | 0.035 |

- The new-feature size-map beats fixed in only **10/50 splits (20%)** — same ballpark as H2's 14%,
  still a clear loss.
- Selected-feature frequency: `gzip_ratio` 23, `paragraph_len_cv` 13, `segment_len_cv` 11,
  `mean_adjacent_cosine` 2, `var_adjacent_cosine` 1 — i.e. the selector keeps reaching for redundancy
  / dispersion signals, and they keep losing. A stably-weak predictor, exactly like TTR before it.

## 5. Verdict (pre-registered)

**❌ DEAD on the cheap, document-intrinsic, training-free route.** Against the plan's gate:

- No feature reaches |ρ| ≥ 0.5 (best is 0.249); none wins ≥ 30/50 splits (best is 10/50).
- **And** the label-derived E2/E3 diagnostics do not explain optimal size (E2 ρ ≈ 0.005).

The second clause is the important one: the pre-registered kill condition is met, so this is not a
"keep searching for a better feature" situation. **Scaling N would not help** — the power analysis
showed N is a limit only when a real signal exists; here even the near-oracle label signal is
absent, so more documents cannot manufacture a correlation that the mechanism doesn't produce.

## 6. What this means for the project

This **strengthens the negative result** decisively over H2:

> On QASPER, per-document leaf-size heterogeneity leaves real oracle headroom (+9.75% rel.), but
> **neither cheap surface density features, nor causally-motivated semantic-locality features, nor
> even the label-derived gold-evidence span length predict per-document optimal leaf size**. The
> optimal leaf size is not an exploitable function of any cheap, document-intrinsic property we
> tested. Practical advice stands: tune one global leaf size (≈200 tok); it is a strong,
> hard-to-beat baseline.

We have now traced the chain end to end and found the break is at the *target* (optimal size is not
set by evidence geometry), not at the *features*. Where a win could still live (future work, honest):
a **query-aware / retrieval-dependent** signal — optimal size may be set by the *question
distribution and retrieval dynamics*, not the document — which is the MoG/HiChunk territory we
deliberately scoped out (it abandons the training-free, document-intrinsic differentiator and is a
larger project).

## 7. Pointers
- Plan: `docs/specs/2026-06-27-weighted-adaptive-chunking-plan.md`
- Features: `raptor/chunking/segment_features.py`; diagnostics: `experiments/granularity_calibration.py`
- H2 (surface) negative result: `docs/results/2026-06-15-h2-density-50doc.md`
- Power analysis / the bar: `docs/results/2026-06-18-signal-recovery-power.md`
- Notebook: `notebooks/locality_calibration.ipynb`
