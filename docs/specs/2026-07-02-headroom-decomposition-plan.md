# Plan — Headroom Decomposition: Where Does Adaptive-Chunking Gain Actually Live?

**Date:** 2026-07-02 · **Branch:** `ckraptor` · **Status:** PROPOSED RESEARCH DIRECTION (new paper)
**Author:** Shagun + research agent
**Builds on:** the negative result in `paper/main.tex` (H1/H2) and the no-LLM harness
(`experiments/granularity_sweep.py`, `experiments/granularity_calibration.py`).
**Trigger:** SLIDERS (Joshi, Shethia, Dao, Lam — *"Contexts are Never Long Enough: Structured
Reasoning for Scalable QA over Long Document Sets"*, arXiv:2604.22294), whose "aggregation
bottleneck" gives us the vocabulary to ask *where* the chunking headroom lives.

---

## 0. TL;DR — what this document is

Our current paper proved a clean **negative**: per-document optimal leaf size is **not** an
exploitable function of any *document-intrinsic* property, so cheap per-document sizing cannot
beat one tuned global size. The paper ends by *conjecturing* the residual signal is
"query-aware / retrieval-dependent" — but never measures it.

This plan turns that conjecture into the **central contribution of a new paper**: a
**decomposition of the oracle headroom** into named, separately-measurable components via
**nested oracles** —

> document-intrinsic · query-type · document×query-type interaction · **within-document /
> per-question (the ceiling)** · aggregation-driven (evidence multiplicity).

The whole adaptive-chunking subfield proposes *methods and reports wins*; **nobody accounts for
where the achievable gain comes from or what the per-component ceiling is.** That accounting is
what our oracle-sweep + positive-control + label-derived-target apparatus is uniquely built to
do. It reframes our negative result as **the first bar of a systematic ledger**, and it makes
SLIDERS' aggregation-bottleneck claim **quantitative** from the retrieval side.

**Cost:** effectively **zero new LLM budget** for the core result. The sweep already performs
all retrieval per question; we currently *throw away* the per-question scores by averaging. We
stop averaging, and the entire decomposition is a re-`argmax` over the same cached records.

---

## 1. Where we are (constraints this plan must respect)

Established, with receipts (`paper/main.tex`, `docs/results/`):

1. **H1 — PASS.** Leaf size matters; per-doc optimum is heterogeneous. QASPER: best fixed 200 tok
   @ 0.7362; per-doc oracle 0.8080 → **+9.8% rel.** headroom.
2. **H2 — FAIL.** No cheap *document-intrinsic* feature (surface density *or* semantic locality)
   predicts optimal size well enough to beat fixed. Best |ρ|=0.38; power analysis says a reliable
   win needs |ρ|≳0.6.
3. **The decisive diagnostic.** Even a **label-derived** target (gold-evidence span length, E2) is
   uncorrelated with optimal size (ρ≈0). ⇒ the optimum is not a function of the *document*.
4. **Generality.** Replicates on QuALITY (2nd corpus, answer-recall proxy): near-parity, no win.
5. **Asset.** A deterministic, resumable, **no-LLM** oracle sweep + a calibration/stat module +
   a synthetic positive-control (`signal_recovery_power.py`). 271 tests passing.

**Hard lesson we must not repeat:** re-weighting document-intrinsic features is already refuted
(the learned regressor lost too). Any new claim MUST move the *conditioning variable* off the
document — onto the **query** — which is exactly what the decomposition does.

---

## 2. The gap in the literature (novelty argument — read before committing)

Query-conditioned granularity as a *method* is **occupied**:
- **Mix-of-Granularity (MoG)** — trained query-conditioned router over pre-segmented levels.
- **Mixture-of-Chunkers (MoC)** — trained router over meta-chunker models.
- **QASC** (arXiv:2605.22834) — query-embedding-driven semantic chunk construction.
- **FreeChunker** (arXiv:2510.20356) — cross-granularity framework.
- **Adaptive-k** (arXiv:2506.08479) — training-free adaptive *retrieval amount*, targets
  aggregation QA.

So "another query-aware granularity method" is a dogfight and, at our scale, a short paper.

**What is NOT occupied:** a principled **accounting of where the headroom lives** and **what the
ceiling is per component**. Every method above reports a benchmark win; none answers *"of the
total oracle gap, how much is capturable by conditioning on the document vs. the query-type vs.
the individual question vs. not capturable at all?"* — nor *"why do these wins fail to transfer
across benchmarks?"* Our answer: **because the exploitable headroom is query-distribution-specific,
not document-intrinsic** — and we can prove it with nested oracles. That is the novel lens, and it
is a *framework* contribution that informs the whole subfield rather than competing inside it.

**SLIDERS bridge:** SLIDERS argues chunk-decomposition dies at the *aggregation* end and you must
switch to structured extraction. We can *measure* the crossover: as gold-evidence multiplicity
rises, the aggregation-attributable headroom grows while the document-intrinsic component stays
~0 — empirically delineating the regime where SLIDERS-style structured reasoning is warranted.
No prior work quantifies that boundary from the retrieval-granularity side.

---

## 3. The core method — nested-oracle headroom decomposition

Let `cov(d, q, S)` = the proxy coverage for question `q` of document `d` at leaf size `S`
(this is the per-question score the sweep *already computes*). Define a ladder of oracles, each
allowed to condition on strictly more:

| # | Oracle | Chooses one size per… | Interpretation |
|---|---|---|---|
| O0 | **Global fixed** | everything (one S) | baseline (the strong baseline we already found) |
| O1 | **Per-document** | document `d` | document-intrinsic headroom |
| O2 | **Per-query-type** | query class `t` (pooled over docs) | query-type-attributable headroom |
| O3 | **Per (doc × query-type)** | `(d, t)` | interaction headroom |
| O4 | **Per-question** | individual `q` | **absolute ceiling** |

**The headline decomposition** (all measured on held-out questions, budget-matched):

```
Total headroom      = cov(O4) − cov(O0)          # everything adaptivity could ever buy
  ├─ Document        = cov(O1) − cov(O0)          # ≈ predictable-0 (our prior result, now a bar)
  ├─ Query-type      = cov(O2) − cov(O0)
  ├─ Interaction     = cov(O3) − cov(O1) − cov(O2) + cov(O0)
  └─ Within-doc/     = cov(O4) − max(O1, O2, O3)  # residual: query-specific, feature-invisible
     instance
```

**The money quantity is `cov(O4) − cov(O1)`** — headroom that exists *within a document, across
its questions*. A document feature is a single number per document; it **cannot** produce
different sizes for different questions of the same document. So this residual is a *hard
information-theoretic ceiling* on any document-intrinsic method — it makes our H2 negative result
**inevitable, not incidental**, and we can put a number on it.

### 3b. Aggregation stratification (the SLIDERS axis)

Stratify questions by **gold-evidence multiplicity** `m(q)` = number of gold-evidence paragraphs
(QASPER ships this per question; free). Buckets: `m=1` (lookup), `m=2` (multi-hop), `m≥3`
(aggregation). Recompute the decomposition **per bucket** and plot each component vs `m`.

**Pre-registered hypotheses:**
- **D1:** the document component stays ≈0 across all `m` (document identity never helps).
- **D2:** total headroom and the within-doc/instance component **grow with `m`** — i.e. the
  harder-to-place evidence is exactly where a fixed size hurts most.
- **D3 (SLIDERS-facing):** for `m≥3`, even the per-question oracle's absolute coverage falls
  (retrieval-granularity tuning cannot rescue aggregation), motivating structured extraction.

Any of D1–D3 is a clean, publishable finding; all three together is a coherent story.

### 3c. Which "query type" taxonomy

Two independent, cheap axes — no LLM required:
1. **Evidence multiplicity** `m(q)` (above) — the SLIDERS axis, always available.
2. **Answer type** from QASPER metadata (extractive / abstractive / yes-no / unanswerable) —
   check the loader exposes it (`experiments/datasets.py`); if not, derive a cheap surrogate
   (answer length, boolean-answer regex). This is the "query-type" for O2/O3.

---

## 4. What has to change in the code (small, no new LLM)

1. **Persist per-question scores in the sweep.** `sweep_document` (line ~257) currently returns
   `mean_evidence_coverage`. Add a `per_question` list: `[{qid, coverage, m, answer_type}]` to
   each `(doc, size)` record. This is the *only* retrieval-touching change and it re-uses the
   retrieval already done — **no new embedding, no new LLM**. Guard behind a flag so existing
   records/tests stay valid; add a one-time re-run to regenerate cached sweeps with the richer
   record (QASPER n=50 sweep is cheap; QuALITY too).
2. **New module `experiments/headroom_decomposition.py`.** Pure-stdlib functions:
   `oracle_global`, `oracle_by_key(records, key_fn)`, `oracle_per_question`, and
   `decompose(records, strata)` returning the ledger in §3. Held-out protocol: reuse
   `granularity_calibration.train_test_split_docs` but split on **questions** for O2/O4 (fit the
   per-key argmax on train questions, score on test) to avoid the optimistic in-sample oracle
   bias. Report mean±std over 50 splits, exactly like H2.
3. **Notebook `notebooks/headroom_decomposition.ipynb`** mirroring the existing notebooks
   (Colab/Drive-cached, resumable) that loads the enriched sweep and prints the ledger + the
   per-`m` component curves + a summary JSON for the writeup.
4. **Tests.** Unit-test the decomposition on a tiny synthetic record set with a *planted* known
   split of headroom (analogous to `signal_recovery_power.py`) so we prove the estimator recovers
   the true component sizes before trusting it on QASPER.

**Explicitly NOT needed for the core result:** any LLM call, any SLIDERS reimplementation, any new
corpus download. The core paper is producible from the two sweeps we already have + the richer
record.

---

## 5. Scale & validity (the honest weak point, and how we address it)

- **N.** QASPER n=50 is our biggest threat for a top venue. The decomposition needs *questions*,
  and we have ~171 (QASPER) + ~448 (QuALITY) — better statistical footing than the 50-doc
  correlations. **Action:** push QASPER to n≈150–200 docs for this paper (sweep cost is modest,
  no LLM). Report the ledger with bootstrap CIs over questions.
- **In-sample oracle bias.** Oracles are optimistic by construction. **Mitigation:** all reported
  oracle coverages use the **held-out** fit/score split (§4.2); the *gaps between* nested held-out
  oracles are the estimand, and the planted-split test (§4.4) validates the estimator.
- **Proxy gap.** Still a retrieval proxy, not end-task QA. **One end-task confirmation** (LLM,
  budgeted, cached) on the *high-`m` stratum only* — the regime our story says matters most —
  would materially strengthen the paper; scope it as an optional Phase C, gated on the proxy
  ledger showing a large within-doc/aggregation component worth confirming.
- **Two-corpus generality is already in hand** (QASPER + QuALITY), so the decomposition can be
  reported on both — the cross-corpus stability of the *shape* of the ledger is itself a result.

---

## 6. Phased execution

- **Phase A — instrument + validate (no LLM).** Code changes §4.1–§4.2 + §4.4 planted-split test.
  Regenerate the QASPER + QuALITY sweeps with per-question records. *Exit:* decomposition
  estimator provably recovers a planted ledger.
- **Phase B — the ledger (no LLM, the core result).** Run the decomposition on both corpora;
  produce the component table, the per-`m` curves, held-out CIs. *Exit:* we can state, with
  numbers, how the +9.8% (QASPER) / +3.0% (QuALITY) headroom splits across document / query-type /
  interaction / within-doc, and how each moves with evidence multiplicity. Write
  `docs/results/2026-07-…-headroom-decomposition.md`.
- **Phase C — optional end-task confirmation (budgeted LLM).** Only if Phase B shows a large
  within-doc/aggregation component: confirm on the high-`m` stratum end-to-end, cached.
- **Phase D — paper.** New paper (or a major-revision superset of the current one): the harness +
  H1/H2 negative become Section "the document axis is empty," and the decomposition + SLIDERS
  bridge become the positive framework contribution.

---

## 7. Why every outcome is publishable (de-risking)

Because this is an **accounting**, not a method, it cannot "fail to beat a baseline":
- If the **within-doc/instance** component is large → we've *quantified why* all document-intrinsic
  methods (ours + the literature's) must plateau, and *located* the real headroom on the query axis
  — a strong, general, cited-by-future-work result.
- If **query-type** headroom is large and cheaply capturable → a training-free query-type→size rule
  that *does* win becomes a bonus method contribution (and cleanly beats MoG/MoC on the
  training-free axis).
- If **aggregation** headroom dominates at high `m` and even O4 coverage collapses → a
  retrieval-side, empirical motivation for SLIDERS-style structured extraction, with a measured
  crossover — novel and directly in dialogue with a current paper.

---

## 8. Risks / open checks (do these first)

1. **Does the QASPER loader expose per-question evidence + answer type?** Verify in
   `experiments/datasets.py` before Phase A; if answer_type is missing, use the evidence-count axis
   alone (sufficient for the SLIDERS story) plus cheap answer-length surrogates.
2. **Per-question held-out oracle variance** at ~3.4 Q/doc may be high. The planted-split test and
   question-level bootstrap CIs are the guardrails; if variance swamps the signal, aggregate query
   *types* rather than individual questions (O2/O3 still stand even if O4 is noisy).
3. **Is the decomposition additive as written?** The interaction term (§3) assumes an ANOVA-style
   split; validate on planted data (§4.4) and, if non-additivity bites, report components as
   *nested marginal gains* (O0→O1→O3→O4 ladder) instead of an orthogonal split — still fully
   interpretable.

---

## 9. Pointers
- Current paper: `paper/main.tex` · Prior briefing: `docs/2026-07-02-progress-report.md`
- Harness: `experiments/granularity_sweep.py`, `experiments/granularity_calibration.py`,
  `experiments/signal_recovery_power.py`
- SLIDERS: arXiv:2604.22294 · code https://github.com/stanford-oval/sliders
- Occupied query-adaptive methods (novelty contrast): MoG (arXiv:2406.00456), QASC
  (arXiv:2605.22834), FreeChunker (arXiv:2510.20356), Adaptive-k (arXiv:2506.08479)
