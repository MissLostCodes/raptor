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

### Integration — *pending H1/H2*
A new chunker/arm (`density` or extend `ahc`) sets each leaf's token budget from the local
density score, respecting structure boundaries where the structure axis fires. Slots into
the frozen RAPTOR pipeline as one more arm; only the chunker varies.

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
- ⏳ density→size calibration (needs E1 output)
- ⏳ adaptive chunker arm + pipeline wiring (needs H1/H2)
- ⏳ E3/E4 end-to-end runs + paper tables
- ⏳ read Mix-of-Granularity (novelty due-diligence)
