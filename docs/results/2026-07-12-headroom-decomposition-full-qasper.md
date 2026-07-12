# Where the headroom lives: nested-oracle decomposition on the full QASPER test split

**Date:** 2026-07-12
**Run:** `notebooks/headroom_decomposition.ipynb` on `ckraptor`, Colab T4, `CORPUS='qasper'`, `N_DOCS=None`.
**Raw summary:** [`2026-07-12-qasper-headroom-summary.json`](./2026-07-12-qasper-headroom-summary.json)
**Scale:** 416 papers · 1,451 answerable questions · 1,352 scored question units · 8,112 (doc, size) cells — the **full QASPER `test` split** (~8× the n=50 cohort used for H1/H2).

## TL;DR

Held out and de-biased, **a per-document leaf-size oracle underperforms a single global size** (O1 = 0.712 vs O0 = 0.756; the `document` component is **−0.045**, 95% band [−0.060, −0.028], firmly below 0). The oracle headroom that does exist (+0.090 abs / +12% rel. at the per-question ceiling) is **almost entirely within-document / per-question** (`within_doc` = +0.115, `money` O4−O1 = +0.135, both clear of 0) — a granularity that *no* document-intrinsic feature can address by construction. This turns the H2 negative result from "features don't predict the optimum" into the stronger, structural "the exploitable signal lives below the document level."

## Method (recap)

Five nested oracles, each allowed one leaf size per *group*, coarse → fine:

| rung | one size per… | meaning |
|---|---|---|
| O0 | everything | best global fixed size |
| O1 | document | per-document oracle (ceiling of any per-doc *feature* policy) |
| O2 | query type (evidence multiplicity `m`) | query-type headroom |
| O3 | document × type | interaction |
| O4 | individual question | absolute ceiling |

To kill the in-sample optimism, rungs are fit on a random 70% of **questions** and scored on the disjoint 30%, over **50 splits** (mean ± 95% band) — the same held-out protocol as H2. All rungs are scored on the same held-out set, so gaps telescope into additive components: `document = O1−O0`, `query_type = O2−O0`, `interaction = O3−O1−O2+O0`, `within_doc = O4−O3`, summing to `total = O4−O0`. O4 is a per-question ceiling (a held-out question was never fit → scored as in-sample best-of-sizes), so `within_doc` is a conservative **upper bound**.

## The ledger (held-out, de-biased — the paper numbers)

| rung / component | mean | 95% band | note |
|---|---|---|---|
| O0 global fixed | 0.756 | [0.735, 0.783] | honest baseline |
| O1 per-document | **0.712** | [0.692, 0.739] | **below O0** |
| O2 per query-type | 0.755 | [0.731, 0.784] | ≈ O0 |
| O3 doc × type | 0.732 | [0.712, 0.762] | below O0 |
| O4 per-question ceiling | 0.847 | [0.830, 0.867] | unreachable ceiling |
| `document` (O1−O0) | **−0.045** | [−0.060, −0.028] | negative, excludes 0 |
| `query_type` (O2−O0) | −0.002 | [−0.014, 0.007] | ≈ 0, spans 0 |
| `interaction` | +0.022 | [0.004, 0.045] | small, excludes 0 |
| `within_doc` (O4−O3) | **+0.115** | [0.097, 0.131] | large, excludes 0 |
| `total` (O4−O0) | +0.090 | [0.073, 0.105] | +12% rel. over global |
| (ref) `money` (O4−O1) | +0.135 | [0.116, 0.152] | headroom no per-doc feature can reach |

Telescoping check: −0.045 − 0.002 + 0.022 + 0.115 = 0.090 = total. ✓

## In-sample vs held-out (the de-biasing)

| component | in-sample (optimistic) | held-out (de-biased) | optimism removed |
|---|---|---|---|
| `document` (O1−O0) | +0.065 | **−0.045** | 0.110 |

Same *total* headroom (~0.090 either way); honest evaluation just relocates it — the document bucket evaporates (goes negative) and everything moves to per-question. The 0.11-point collapse in O1 is pure overfitting exposed: with only ≈3.3 questions/paper, a per-doc size fit on ~2 questions and applied to ~1 held-out question tracks noise and does worse than the safe global default.

## The aggregation axis (SLIDERS / evidence multiplicity `m`)

Stratified by `m` (in-sample; magnitudes are conservative, the trend is the signal):

| stratum | O0 (global-size coverage) | within-doc (O4−O1, in-sample) |
|---|---|---|
| m=1 (lookup) | 0.832 | 0.004 |
| m=2 (multi-hop) | 0.752 | 0.010 |
| m≥3 (aggregation) | 0.698 | 0.019 |

- **O0 falls monotonically** 0.83 → 0.75 → 0.70 — a *direct measurement* (not an oracle): aggregation is retrieval-hard regardless of tuning.
- **within-doc headroom grows with `m`** — the exploitable signal concentrates exactly where retrieval is hardest and a single leaf size is least adequate.

## Interpretation for the paper

1. **Stronger negative result.** The per-document oracle is the *ceiling* of any per-document-feature policy. If even the oracle loses to global out-of-sample, the entire per-document adaptation family is dominated — a sharper claim than H2's "no feature predicts the optimum."
2. **Structural, not incidental.** A per-document feature emits one number per document and cannot vary size across a document's questions; the `within_doc` component is a hard ceiling it can never reach. The signal lives at the individual-question level → the H2 failure was inevitable.
3. **Direct evidence for the "query-aware" conjecture.** The paper's Discussion previously *conjectured* the residual signal is query-aware / retrieval-dependent; the decomposition now shows it directly (within-question, and concentrated in high-`m` aggregation).

## Honest caveats

- QASPER gives ~3.3 questions/paper, so O1 is fit on ~2 questions — data-starved by construction. The defensible claim is operational: *at realistic per-document question counts, per-document size selection can't be fit reliably and underperforms the global default.*
- Per-`m` magnitudes are in-sample (cell 6 is not held-out); only the O0-fall (direct) and the growth trend are load-bearing.
- Still a retrieval-coverage proxy, not end-task QA (same threat as the rest of the paper).

## Cross-corpus replication: QuALITY

Same notebook, `CORPUS='quality'`, full validation split — **115 articles, 2,086 questions** (answer-recall proxy, so `m≡0` and the query-type/aggregation axes are inert). Raw: [`2026-07-12-quality-headroom-summary.json`](./2026-07-12-quality-headroom-summary.json).

| component (held-out) | QASPER | QuALITY |
|---|---|---|
| `document` (O1−O0) | −0.045 [−0.060, −0.028] | **−0.003 [−0.010, +0.005]** (spans 0) |
| `within_doc` (≈ O4−O1) | +0.115 / +0.135 | **+0.046 [0.040, 0.052]** |
| `total` (O4−O0) | +0.090 | +0.043 |
| questions / doc | ~3.3 | ~18 |

**The shape replicates.** On both corpora, held-out per-document adaptation does **not** beat a global size (`document ≤ 0`), and essentially **all** headroom is within-document (`within_doc` ≈ the entire `total`).

**The magnitude difference is explained by data-per-doc, not corpus.** QASPER's `document` is *firmly negative* (−0.045) because ~3.3 questions/paper starves the per-doc fit → it overfits and actively hurts. QuALITY's ~18 questions/article give enough data to fit a per-doc size *without* that penalty, so it lands at parity (−0.003) — but the extra data buys parity, never a win. Either way, no exploitable per-document signal exists.

Caveat: QuALITY's best fixed size hits the grid ceiling (400 tokens), so its absolute coverage may be mildly under-resolved; the decomposition compares oracles on the same grid and is unaffected.

## Next

- **Paper integration: done** (2026-07-12) — new §"Where the Headroom Lives" (QASPER ledger + QuALITY replication paragraph), abstract, Discussion, and Threats updated.
- Open: verify the SLIDERS arXiv id (`\todo` in `sec:headroom`) and add its `\bibitem`.
- Optional Phase C: one budgeted LLM end-task confirmation on the high-`m` aggregation stratum (QASPER only), to close the retrieval-proxy vs end-task gap.
