# Density-Adaptive Leaf Granularity for Hierarchical RAG — Research Spec

**Date:** 2026-06-11 · **Branch:** `ckraptor` · **Status:** active (de-risk gate pending)

## One-line contribution

A **training-free, deterministic, two-axis chunking controller** for hierarchical
(RAPTOR-style) retrieval: a **structure score** decides *where* to cut (boundaries /
route), and a separate **density score** decides *how big* each leaf should be
(granularity). The density→size map is **calibrated against an oracle leaf-size sweep**
using QASPER gold-evidence retrieval — no learned model, interpretable, and competitive
with a learned granularity selector at ~zero cost.

## Motivation & the gap

Fixed leaf size (RAPTOR uses 100 tokens) ignores that documents differ in information
density. The *broad* idea "adapt chunk size to content" is **not** novel on its own:

- **Mix-of-Granularity** (arXiv:2406.00456) — *learns* to optimize chunk granularity (a
  trained soft router). This is the closest prior art and the "use a model" route we
  deliberately avoid as a contribution.
- **HiChunk + Auto-Merge** (arXiv:2509.11552) — hierarchical structuring via a fine-tuned
  LLM, query-adaptive granularity.
- **"Rethinking Chunk Size for Long-Document Retrieval"** (arXiv:2505.21700) — chunk-size
  analysis; no per-document policy.
- **Adaptive Chunking: chunking-method selection** (`ekimetrics/adaptive-chunking`) — a
  classifier picks a *method*, not a continuous *granularity*.
- Practitioner "adaptive density" heuristics (smaller chunks for dense paragraphs) are
  common but **never formalized or validated**.

Recurring refrain across the literature: *"there is no universal formula; optimal chunk
size is empirical."* **The unoccupied niche:** a *cheap, closed-form, training-free* score
that sets **per-section leaf granularity** in a **hierarchical tree** and is *shown to
match an oracle* and *rival a learned selector*. Differentiators that survive review:
training-free vs Mix-of-Granularity · interpretable closed form (you can read off why a
section got its size) · hierarchical-leaf-specific · validated against an oracle, not
hand-tuned.

**Conceptual novelty:** *decompose chunking control into two cheap deterministic axes —
structure→boundary and density→size — rather than one learned granularity selector.* No
prior work states this decomposition cleanly.

## Hypotheses

- **H1 (de-risk gate) — size matters & is heterogeneous.** On QASPER, leaf granularity
  materially changes gold-evidence retrieval, **and** the optimal granularity varies
  across documents. *If false (flat curve or one shared optimum), there is no adaptive
  contribution — we stop and report the negative result + best fixed size.*
- **H2 — cheap features predict it.** Closed-form density features predict per-document
  optimal granularity better than a global constant, approaching a learned regressor on
  the same features.
- **H3 — end-to-end win.** Density-adaptive leaf sizing improves QASPER Answer-F1 /
  evidence-F1 over fixed-100 and over best-global-fixed, approaching the per-doc oracle —
  at zero training and negligible cost.
- **H4 — generality / no harm.** On unstructured corpora (QuALITY narrative), adaptive
  sizing does no harm (the structure axis correctly avoids over-segmentation).

## Method

### Axis 1 — Structure score (boundaries / routing) — *implemented*
`raptor/chunking/structure_score.py`. Deterministic surface features (numbered/markdown/
ALL-CAPS/keyword headings, TOC, spacing) **plus a generic Title-Case heading detector**
(added 2026-06-11) so plain title-case section headers (QASPER, reports, wikis) are
recognized without false-firing on prose. Validated: QASPER-style → ≥0.70 → structure;
QuALITY prose → 0.04 → token.

### Axis 2 — Density score (granularity) — *implemented (extractor); calibration pending*
`raptor/chunking/density_score.py`. `density_score(text) -> (rho∈[0,1], features)` from six
cheap features: mean sentence length, type-token ratio, numeral/symbol density, mean word
length, list-marker density, non-stopword ratio. `density_to_leaf_tokens(rho, l_min, l_max,
invert)` maps ρ→leaf budget. **Weights, direction (`invert`), and bounds are calibration
parameters with placeholder defaults — fixed by the oracle sweep, not assumed.**

### Calibration — oracle leaf-size sweep w/ gold-evidence proxy — *experiment built*
`experiments/granularity_sweep.py` + `notebooks/granularity_sweep.ipynb`. Cheap (SBERT
only, **no LLM**): for each QASPER doc × leaf size S ∈ {50,100,150,200,300,400}, build a
flat SBERT+FAISS leaf index (reusing the `flat`-arm machinery), retrieve under a 2000-token
budget, and measure **gold-evidence coverage** (token-recall of each gold paragraph within
the retrieved context ≥ 0.8 — *not* `evidence_f1`, which is exact-paragraph set overlap and
incompatible with token chunks). Yields `optimal_size(doc) = argmax_S coverage`. The
density formula (few params, train/test split, no test tuning) is then fit to predict
`optimal_size` from density features.

### Integration — *built (mechanism); awaiting calibrated coefficients from H2*
`raptor/chunking/density_adaptive_chunker.py` (`DensityAdaptiveChunker`) is the `density`
arm: a two-axis controller that routes on the structure score (structure vs token, like
AHC) **and** sets each document's leaf token budget from an injected `size_fn`. The
calibrated policy is built by `make_calibrated_size_fn(feature, a, b, l_min, l_max)` from
the `select_and_fit` output and bridges offline calibration → online chunking (needs only
the one density feature at inference — no SBERT/LLM/labels). Wired through
`get_chunker("density", size_fn=…)`, `ExperimentConfig.density_{feature,a,b,l_min,l_max}`,
`pipeline.make_ra_config` (`TREE_ARMS` now includes `density`), and `routing_info` (logs
the per-document `leaf_size`). Default `size_fn` is the **uncalibrated composite
placeholder** — runnable but not for reported numbers. The mechanism is unit-tested
offline; only the fitted `(feature, a, b)` are pending the ~50-doc H2 run.

## Experiments

| ID | Question | Cost | Tests |
|---|---|---|---|
| **E1** | Does leaf size move gold-evidence coverage, and is the optimum doc-dependent? | cheap (SBERT only) | H1 (gate) |
| **E2** | Do density features predict optimal size? vs global-constant & learned regressor | cheap | H2 |
| **E3** | Full RAPTOR+QA: fixed-100 · best-global-fixed · **density-adaptive (ours)** · oracle-size (upper bound) · semantic | expensive (LLM) | H3 |
| **E4** | E3 on QuALITY / NarrativeQA — no-harm & generality | expensive | H4 |

**Baselines:** RAPTOR fixed-100, best-global-fixed (oracle-fixed), semantic chunking, and a
**learned regressor on the same density features** (to show training-free ≈ learned).
**Upper bound:** per-doc oracle size. **Metrics:** QASPER Answer-F1 + evidence-F1;
QuALITY accuracy; NarrativeQA ROUGE-L/BLEU/METEOR.

## Constraints (fixed by hardware/budget)

- **Compute:** free Colab (T4), session disconnects → all runs resumable + Drive-cached.
- **Models:** free open-source only — `gpt-oss-120b` (Cerebras free, summary+QA) and SBERT
  (`multi-qa-mpnet-base-cos-v1`) embeddings. No paid APIs.
- **Implication:** small N per dataset, wide CIs; the cheap SBERT-only de-risk (E1/E2)
  carries most of the inferential weight; LLM runs (E3/E4) kept minimal and cached.

## Threats to validity

- Small N (free tier) → wide CIs; lean on the cheap evidence-coverage proxy for power.
- Proxy gap: evidence-coverage (retrieval) vs end-task QA — mitigated by E3 confirming
  end-to-end.
- Overfitting the formula → few parameters + held-out test split, no test tuning.
- Fixed embedder/reader: results are within-pipeline (arm-vs-arm), not leaderboard numbers.
- **Novelty due-diligence (must-do):** read Mix-of-Granularity in full to confirm it is
  *learned* (not closed-form) and does **not** cover the hierarchical-leaf case.

## Paper artifacts this produces

- Fig. 1: evidence-coverage vs leaf size (aggregate) — "size matters."
- Fig. 2: per-doc coverage curves / argmax-size distribution — "optimum is heterogeneous."
- Table 1: predicted-optimal vs oracle vs global-fixed vs learned (H2).
- Table 2: end-to-end QASPER (+ QuALITY/NarrativeQA) arms vs oracle upper bound (H3/H4).
- Feature ablation: which density features carry the signal.

## Phasing & decision gates

1. **Phase 1 (done):** generic structure detector, QASPER token-vs-ahc notebook, QuALITY
   pilot results doc.
2. **Phase 2a (built, awaiting run):** E1 de-risk sweep (`granularity_sweep.ipynb`).
   **GATE:** H1 must hold or we pivot.
3. **Phase 2b (post-gate):** fit density→size; E2; integrate the adaptive arm.
4. **Phase 2c:** E3/E4 end-to-end; assemble paper tables/figures.

## Implementation status

- ✅ `structure_score.py` generic Title-Case detector (+ tests)
- ✅ `density_score.py` extractor + `density_to_leaf_tokens` (+ tests)
- ✅ `experiments/granularity_sweep.py` + tests + `notebooks/granularity_sweep.ipynb`
- ✅ `notebooks/qasper_ahc_vs_token.ipynb`, `docs/results/2026-06-11-quality-ahc-vs-token-pilot.md`
- ✅ H1 de-risk gate PASSED (8-doc pilot: best size spans 50→400, doc-heterogeneous)
- ✅ `experiments/granularity_calibration.py` (headroom accounting + closed-form
  density→size fit, train/test split) + 30 tests + `notebooks/granularity_calibration.ipynb`
  (scales sweep to ~50 docs, headroom Fig.1, Table 1 held-out eval, learned-regressor baseline)
- ✅ novelty due-diligence memo `docs/specs/2026-06-11-related-work-novelty.md`
  (closest threat = Ekimetrics 2603.25333, not Mix-of-Granularity)
- ✅ H2 PREVIEW (n=8, `docs/results/2026-06-11-h2-density-feature-preview.md`): the
  equal-weight composite `rho` is near-zero predictive (+0.04) because features are
  strongly but oppositely signed (`mean_word_len` +0.79, `nonstopword_ratio` +0.66,
  `type_token_ratio` −0.60). Method updated: `granularity_calibration.select_and_fit`
  picks the strongest single feature on TRAIN (closed-form, 1 feat + 2 params) instead
  of the dead composite; calibration notebook reports it as the headline training-free arm.
- ✅ adaptive chunker arm + pipeline wiring: `DensityAdaptiveChunker`, `make_calibrated_size_fn`,
  `get_chunker("density")`, config + `make_ra_config` + `routing_info` (mechanism unit-tested,
  271 suite-wide). Only the fitted `(feature, a, b)` are pending H2.
- ⏳ run the scaled (~50-doc) sweep + calibration notebook → H2 gate (selected-feature
  closed-form beats best-fixed, approaches learned regressor + oracle on held-out docs);
  then drop `(feature, a, b)` into `ExperimentConfig.density_*` and run E3.
- ⏳ E3/E4 end-to-end runs + paper tables
- ⏳ full read of Ekimetrics 2603.25333 before camera-ready
