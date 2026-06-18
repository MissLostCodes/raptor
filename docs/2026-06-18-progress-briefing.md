# Research Progress Briefing — Adaptive Chunking for Hierarchical RAG

**Prepared for:** research-guide meeting · **Date:** 2026-06-18 · **Branch:** `ckraptor`
**Student:** Shagun Gupta

This is a one-page-style status you can read top-to-bottom in the meeting. It says **what we
are studying**, **what we built**, **what the numbers say so far**, and **what is next**.
Every number below is traceable to a committed result doc in `docs/results/`.

---

## 1. What we are studying (the research question)

RAPTOR (Sarthi et al., 2024, arXiv:2401.18059) builds a hierarchical retrieval tree by
recursively clustering and LLM-summarizing **leaf chunks**. It inherits a **naive, fixed
leaf chunker** (≈100 tokens, same size for every document). Our question:

> Can we set leaf chunking **per-document**, **cheaply**, and **without training a model**,
> and does that beat a single well-tuned fixed setting inside a frozen RAPTOR pipeline?

We frame chunking control as **two cheap, deterministic axes** (this decomposition is the
conceptual hook of the project — no prior work states it cleanly):

| Axis | Signal | Decides | Status |
|---|---|---|---|
| **Structure** | `structure_score(text)` (surface heading/TOC features, no LLM) | **WHERE to cut** (boundaries / route to structure-aware vs token chunking) | implemented + tested |
| **Density** | `density_score(text)` (6 surface features, no LLM) | **HOW BIG** each leaf is (granularity) | implemented; **calibration refuted** (see §4) |

The validity argument throughout: **only the chunker changes** — identical SBERT embeddings,
clustering, summarizer, QA model, and retrieval config across arms.

---

## 2. What we built (infrastructure — this is solid and reusable)

1. **A training-free, no-LLM evaluation harness** for leaf granularity in hierarchical RAG:
   - `experiments/granularity_sweep.py` — for each QASPER doc × leaf size
     S ∈ {50,100,150,200,300,400}, builds a flat SBERT+FAISS leaf index, retrieves under a
     2000-token budget, and measures **gold-evidence coverage** (token-recall ≥ 0.8 of each
     gold-evidence paragraph). **No LLM anywhere** → cheap, resumable, Colab/Drive-cached.
   - `experiments/granularity_calibration.py` — headroom accounting, per-feature Spearman
     ablation, closed-form `size = a + b·feature` fit, train/test split, 50× repeated-split
     robustness. Pure stdlib, fully unit-tested.
2. **The two-axis controller, wired as a pipeline arm:**
   - `structure_score.py` (+ generic Title-Case heading detector so QASPER sections are
     recognized), `density_score.py`, `DensityAdaptiveChunker`, `make_calibrated_size_fn`,
     and `get_chunker("density", …)` integrated through `ExperimentConfig` / `make_ra_config`
     / `routing_info`. **271 tests passing.**
3. **A reproducible experiment runner**: pinned eval subsets, resumable runs, paired
   bootstrap significance tests, cost/compute accounting, free-tier (Cerebras `gpt-oss-120b`
   + SBERT `multi-qa-mpnet-base-cos-v1`) — no paid APIs.

**Why this matters for the meeting:** the harness is the durable contribution. It lets us
ask "does adaptive granularity help?" *without* paying LLM cost, which is what made a clean
negative result affordable on a free-tier budget.

---

## 3. Result H1 — leaf size matters, and the optimum is heterogeneous → **PASS (n=50)**

*Source: `docs/results/2026-06-15-h2-density-50doc.md` (50 QASPER papers, 171 answerable
questions, 300 (doc,size) records).*

- **Real headroom exists.** Best single fixed size = **200 tok @ coverage 0.7362**;
  per-document **oracle = 0.8080**; absolute gap **0.0718** → **+9.8% relative** gold-evidence
  coverage left on the table by any fixed size.
- **The optimum genuinely varies across documents.** Per-doc best-size distribution over 50
  papers: `{50: 17, 100: 11, 150: 6, 200: 8, 300: 3, 400: 5}` — mass at every size.

H1 is the **premise**, not the contribution: it says an adaptive method *could* win. It
replicates the 8-doc pilot and holds at scale.

---

## 4. Result H2 — can cheap features capture that headroom? → **FAIL (n=50)** (our honest negative result)

*Same source.* This is the scientifically important finding.

**Per-feature Spearman vs per-doc optimal size (n=50):** the strongest is
`type_token_ratio` at only **−0.379** (moderate); all others |ρ| ≤ 0.16; the equal-weight
composite `rho` is −0.147. The features that looked strong on the 8-doc preview
(`mean_word_len` +0.79, `nonstopword_ratio` +0.66) **collapsed** at n=50 → that was
small-N overfitting.

**Held-out robustness over 50 random 70/30 splits (the headline statistic):**

| Policy | mean TEST coverage | std |
|---|---:|---:|
| **best global fixed (200 tok)** | **0.7291** | 0.0433 |
| training-free: composite `rho` → size | 0.7212 | 0.0447 |
| training-free: **selected feature** → size | 0.7037 | 0.0494 |
| learned regressor (sklearn, 6 features) | 0.7071 | 0.0415 |
| per-doc oracle (ceiling) | 0.8061 | 0.0347 |

- Selected-feature map beats the tuned fixed size in only **7/50 splits (14%)**.
- The selected feature **is** stable (`type_token_ratio` chosen in 47/50 splits) — so this is
  not noise; it is a **stably weak** predictor.
- **The learned regressor also loses** (0.707 < 0.729). The bottleneck is the **feature set**,
  not the closed-form mapping — no amount of model sophistication on these cheap density
  features converts the oracle headroom into a win.

**Verdict (against the pre-registered gate in the spec):** of the three gate criteria —
(i) high win-rate over fixed, (ii) stable feature, (iii) approaches learned *and* oracle —
**only (ii) is met. H2 GATE FAILED.**

> **The takeaway:** On QASPER, per-document leaf-size heterogeneity creates real oracle
> headroom (+9.8% rel.), **but cheap training-free density features — and a learned
> regressor on them — do not predict per-document optimal leaf size well enough to beat a
> single tuned global fixed size.** The widely-repeated "smaller chunks for denser text"
> heuristic is **not supported** at the hierarchical-leaf level. Practical advice: tune one
> global leaf size (≈200 tok here); it is a strong, hard-to-beat baseline.

This is a **publishable, honest result**: a first clean formalization *and refutation* of a
common-but-unvalidated heuristic, carried entirely by the cheap proxy (no expensive LLM run
needed).

---

## 5. Result (structure axis) — AHC does no harm on unstructured text → control confirmed

*Source: `docs/results/2026-06-11-quality-ahc-vs-token-pilot.md` (QuALITY, 6 docs, 105
paired questions).* AHC vs token: **statistical tie** (0.400 vs 0.381, overlapping CIs)
because the **structure-routing rate was 0.0000** — QuALITY is plain prose, so AHC correctly
fell back to token chunking on every doc. This is the intended **unstructured control**: AHC
does no harm when there is no structure to exploit. The structure path is genuinely exercised
only on QASPER (`notebooks/qasper_ahc_vs_token.ipynb` — run not yet recorded as a result).

---

## 6. Decisions made (and why)

- **Pivoted the density thread to a negative result** per the pre-registered gate: we do
  **not** inject `(feature, a, b)` into the pipeline and do **not** run the expensive E3
  end-to-end LLM eval for the density arm — the proxy already shows it loses to fixed.
- **Method correction (preview → scale):** the equal-weight composite density score cancels
  signal (features are strongly but oppositely signed). We switched to selecting the single
  strongest feature on the train split — and even that fails at n=50. Documenting this
  preview→scale collapse is itself a methodological honesty point.

---

## 7. What I'd like to discuss / next steps

1. **Framing:** position the paper as **harness + H1 headroom + H2 negative result** (a
   cautionary, reproducible refutation), not as a "we beat the baseline" paper. Negative
   results on a popular heuristic are valuable — does the guide agree this is the right frame?
2. **Strengthening the negative result:** add a second corpus (NarrativeQA / QuALITY) to show
   the refutation is not QASPER-specific; widen CIs are the main threat (N=50, single corpus).
3. **Where a win could still live (future work, honest):** the oracle headroom is real and
   *unclaimed* — a *richer / query-aware* signal (à la Mix-of-Granularity, HiChunk) might
   capture it. We only refute the **cheap, closed-form, training-free density** route, which
   was the intended contribution.
4. **Open question for the guide:** is the **two-axis decomposition** (structure→boundary,
   density→size) worth foregrounding as the conceptual contribution even though the density
   axis returned negative — or do we lead with the harness?

---

## 8. Honesty / limits (state these up front)

- Within-pipeline **proxy** metric (gold-evidence token-coverage), not end-task QA — arm-vs-arm,
  not leaderboard numbers.
- **N=50, single corpus (QASPER)**; wide CIs (std ≈ 0.04). The negative conclusion is robust
  *for the cheap-density route* precisely because the learned upper bound on the same features
  also fails — but it does not generalize to other doc types or richer signals.
- Free-tier nondeterminism (clustering + LLM) → reported with variance; all LLM calls cached.

---

### Pointers (for anyone who wants the receipts)
- H1/H2 final: `docs/results/2026-06-15-h2-density-50doc.md`
- H2 8-doc preview (overfit warning): `docs/results/2026-06-11-h2-density-feature-preview.md`
- QuALITY AHC control: `docs/results/2026-06-11-quality-ahc-vs-token-pilot.md`
- Density spec + gate: `docs/specs/2026-06-11-density-adaptive-chunking.md`
- Novelty / related-work memo: `docs/specs/2026-06-11-related-work-novelty.md`
- Paper draft: `paper/main.tex`
