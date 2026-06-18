# Brainstorm — richer document features & a weighted strategy for optimal leaf size

**Date:** 2026-06-18 · **Branch:** `ckraptor` · **Status:** LIVING brainstorm (not a spec yet).
**Author:** Shagun + research agent. **Purpose:** find document features that could predict
per-document optimal leaf chunk size *well enough to beat a tuned fixed size* — the thing the
6 cheap surface features (H2) failed to do — and design an honest "weighted" combination.

> This doc is deliberately critical, not a wishlist. Every idea is checked against what we
> already **proved**, and against a hard quantitative **bar** our power analysis set.

---

## 0. What we already know (the constraints any new idea must respect)

1. **H2 (n=50) refuted the 6 cheap surface features.** Selected-feature map and the equal-weight
   composite both lose to fixed=200 tok (`docs/results/2026-06-15-h2-density-50doc.md`).
2. **A weighted linear combo is ALREADY tested — and lost.** The "learned 6-feature regressor"
   *is* an optimally-weighted linear combination of those features (it fits the weights to
   minimize error). It scored 0.707 < 0.729. **So hand-tuning weights on the same features
   cannot win** — least-squares already found the best weights and they weren't enough. Any new
   "weighted strategy" must therefore bring **new features** or a **non-linear** combiner, not
   re-weight the old six.
3. **The bar (from `docs/results/2026-06-18-signal-recovery-power.md`).** A *clean* signal needs
   correlation **|ρ| ≳ 0.6** with optimal size to reliably (≥80% of splits) beat fixed. Our best
   single feature reached only 0.38, and even a clean 0.38 wins ~half the splits — QASPER's real
   0.38 won 14%. **Target for a new feature: |ρ| ≳ 0.5–0.6 on held-out docs.**
4. **The headroom is real and unclaimed.** Oracle beats fixed by **+9.8% rel.** coverage. The
   signal exists in the documents; the *features* just don't see it.
5. **Build-time, per-document, no-LLM, free-tier.** RAPTOR fixes leaf size when it builds the tree,
   *before any query is known*. So a deployable feature must be computable from the **document
   alone**, cheaply (SBERT is already in the pipeline; spaCy/gzip are cheap; no paid APIs).

---

## 1. The reframe: optimal leaf size is about EVIDENCE LOCALITY, not "text density"

Why did surface density fail? Because **"dense text" is not the thing that sets the best chunk
size.** The best leaf size is set by **how the answer-bearing information is distributed** in the
document:

- If a document's relevant facts sit in **short, self-contained spans** → **small leaves** win
  (a big chunk dilutes the evidence with irrelevant tokens, hurting retrieval precision).
- If relevant facts are **spread across a coherent passage** that only makes sense together →
  **large leaves** win (a small chunk fragments the evidence and loses recall).

Type-token ratio, word length, etc. are only *very* loose proxies for this. The features most
likely to clear the bar are ones that measure **the document's natural semantic segment size and
how sharply topics shift** — i.e., the size of the largest coherent unit. This is the central
hypothesis of this brainstorm:

> **H-new:** *The optimal leaf size of a document tracks its intrinsic **semantic segment size** —
> measurable cheaply from sentence-embedding coherence — far better than surface lexical density.*

This is also why MoG (arXiv:2406.00456) and HiChunk (arXiv:2509.11552) use **learned / semantic**
signals, not surface stats. Our differentiator stays the same (training-free, hierarchical-leaf,
oracle-calibrated) — we just feed it features that can actually carry the signal.

---

## 2. Candidate feature classes, tiered by (expected signal × cheapness)

Legend: **Cost** = compute on free Colab T4. **Why it could work** = causal link to leaf size.
**Risk** = why it might still be dead.

### TIER A — Semantic-coherence features (BEST BET: cheap *and* causally on-target)
We already embed every sentence with SBERT in the sweep, so these are nearly free.

| # | Feature | What it measures | Why it could set size | Cost |
|---|---|---|---|---|
| A1 | **Mean adjacent-sentence cosine** | local topical continuity | high continuity → big coherent chunks OK → larger leaves | ~free |
| A2 | **Variance of adjacent-sentence cosine** | how "lumpy" topic flow is | high variance → frequent shifts → smaller leaves | ~free |
| A3 | **Topic-shift rate** (# of TextTiling/embedding boundaries per 1k tokens) | density of semantic breaks | more breaks → smaller natural segments → smaller leaves | cheap |
| A4 | **Mean semantic-segment length** (tokens between detected boundaries) | the document's *natural* chunk size | **direct estimate of optimal leaf size** | cheap |
| A5 | **Std/CV of semantic-segment length** | heterogeneity of segments | guides whether one size fits the doc at all | cheap |
| A6 | **Embedding self-similarity / redundancy** (mean pairwise cosine of a sentence sample) | global repetitiveness | redundant docs tolerate bigger chunks | cheap |
| A7 | **Local intrinsic dimensionality / anisotropy** of sentence vectors | semantic spread | higher spread → finer chunks | cheap |

**A4 is the headline candidate**: it is a *training-free, per-document estimate of the natural
segment size itself* — almost definitionally the right scale for a leaf. If anything clears the
bar, it is most likely A3/A4.

### TIER B — Structural / layout statistics (cheap, already partially parsed)
We already compute `structure_score`; these are its raw ingredients as *sizing* features.

| # | Feature | Why it could set size | Cost |
|---|---|---|---|
| B1 | **Paragraph length distribution** (mean, median, CV of tokens/paragraph) | paragraphs are author-intended units; their size ≈ a good leaf size | ~free |
| B2 | **Sentences per paragraph** | denser packing → finer leaves | ~free |
| B3 | **Section length distribution** (tokens between headings) | section size bounds useful leaf size | cheap |
| B4 | **Heading density** (headings per 1k tokens) | more sub-structure → smaller natural units | ~free |
| B5 | **List/table fraction** | enumerated facts → tiny leaves | ~free (have it) |

**B1 (paragraph size)** is the cheap structural analogue of A4 and a strong, interpretable
candidate. Worth testing right next to A4.

### TIER C — Richer linguistic features (cheap-ish, no LLM; spaCy/gzip)
| # | Feature | Why it could set size | Cost |
|---|---|---|---|
| C1 | **Named-entity density** (entities per 100 tokens, spaCy) | fact-dense text → small precise leaves | medium (spaCy) |
| C2 | **Noun-phrase / content-chunk density** | information units per span | medium |
| C3 | **Syntactic complexity** (mean dependency-tree depth / parse length) | complex prose → coherence needs bigger chunks | medium |
| C4 | **Compressibility (gzip ratio)** | model-free redundancy proxy; low redundancy → finer leaves | ~free |
| C5 | **Discourse-marker density** ("however", "therefore", …) | argumentative flow → bigger coherent units | ~free |
| C6 | **Coreference chain span** (avg distance between mentions) | long chains → context must stay together → larger leaves | expensive |

C4 (gzip) is a delightfully cheap, model-free redundancy signal worth including as a baseline.
C1 (entity density) is the classic "information density" done *properly* (vs our token heuristic).

### TIER D — Query-aware features (strongest in the literature, but NOT build-time available)
| # | Feature | Why it could set size | Catch |
|---|---|---|---|
| D1 | Expected answer length / type (factoid vs explanatory) | short answers → small leaves | needs the query; **unknown at tree-build time** |
| D2 | Question specificity / entity overlap | — | same catch |

**Honest limitation:** RAPTOR sizes leaves once, per document, *before queries exist*. Genuine
query-adaptivity (MoG/HiChunk territory) needs an assumed query workload. Useful only if we
either (a) assume a per-corpus query distribution, or (b) move to query-time re-chunking (a
different, bigger project). **Keep D out of the cheap deployable scope; note as future work.**

### TIER E — Target-side diagnostics (NOT deployable features — they tell us what to predict)
Compute from the oracle sweep + gold evidence to learn *which* document properties the optimal
size actually responds to (so we stop guessing):
- **E1 Oracle coverage-curve shape per doc**: peak sharpness, flatness, and *where* the peak is.
  Flat curve → size doesn't matter for that doc (a confound that depresses correlations).
- **E2 Gold-evidence span length distribution** (tokens per gold paragraph). **This is likely the
  true latent driver of optimal size** — if we can predict E2 from A/B/C features, we win.
- **E3 Evidence dispersion**: are a question's gold spans contiguous or scattered across the doc?

**Use E to supervise the search**: regress optimal size on E2/E3 first. If even *those* (near-oracle,
label-derived) don't explain optimal size, the target is too noisy and no document feature will —
that closes the project honestly. If they do, find the cheap A/B/C feature that best predicts them.

---

## 3. The "weighted strategy" — done honestly

Your idea: weight different document properties and combine them into one optimal-size predictor.
Three ways to instantiate it, in increasing promise:

1. **Linear weighted sum (❌ already refuted).** `size = Σ wᵢ·featureᵢ`. With learned weights this
   is exactly the regressor that lost. **Do not ship this on the old features.** Only worth
   revisiting once Tier-A/B features are in the pool — and even then expect a non-linear combiner
   to be needed.
2. **Non-linear / interaction combiner (maybe).** A shallow gradient-boosted tree or a 2-feature
   interaction (e.g., `segment_size × coherence_variance`). Captures "big leaves only when BOTH
   segments are long AND topic flow is smooth." Risk: 50 docs is tiny → overfitting; must use the
   held-out 50-split protocol and keep ≤2–3 features.
3. **Physically-motivated direct estimator (BEST).** Skip abstract weighting; set
   `leaf_size ≈ natural_segment_size(doc)` from A4/B1 directly, then a 1-parameter global scale
   fit on TRAIN. The "weights" become a tiny, interpretable calibration (slope + clip), exactly
   like our current closed form but on a feature that's *causally* the right quantity. This is the
   cleanest paper story and the least overfit.

**Recommended weighting design:** a **2-stage, ≤3-feature, monotone** model —
   - *Stage 1 (regime):* is this doc size-sensitive at all? (from coherence variance A2/A5).
   - *Stage 2 (size):* predict size from the natural-segment-size feature (A4 or B1), 1-param
     global calibration; fall back to fixed=200 when Stage 1 says "flat curve."
   Keep it interpretable and hard to overfit. (Note: our earlier synthetic test showed *hard*
   step rules are risky — so make Stage-2 a smooth calibrated map, not a discrete jump.)

---

## 4. How to test cheaply — reuse the harness we already built (no new LLM cost)

We do **not** need new experiments infrastructure. The oracle sweep already gives per-doc optimal
size, and we have the calibration + power tooling. The loop per candidate feature:

1. Compute the feature for all 50 QASPER docs (Tier A/B from sentences we already embed; C from
   spaCy/gzip). Add to a `feats` dict exactly like `density_score`'s output.
2. **Spearman vs optimal size** (`gc.rank_corr`) — first gate: is |ρ| ≳ 0.5? (cf. old best 0.38).
3. **Held-out 50-split eval** (`select_and_fit` → `coverage_under_sizes`) — does it beat fixed,
   and at what win-rate?
4. **Power check** — compare its win-rate to the clean-signal curve in `signal_recovery_power.py`.
5. **Predict-the-driver check (E2):** does the feature predict gold-evidence span length?

**Pre-registered success / kill criteria (set now, before looking):**
- ✅ *Promising* if a single new feature reaches **|ρ| ≥ 0.5 held-out** AND its sizing map beats
  fixed in **≥ 30/50 splits**.
- ⚠️ *Marginal* if 0.4 ≤ |ρ| < 0.5 — only pursue via the 2-feature interaction.
- ❌ *Dead* if even the **label-derived E2/E3** diagnostics don't explain optimal size — then the
  per-doc target is irreducibly noisy at this N, and we **stop and keep the negative result**.

All of this is **SBERT-only / CPU-cheap, resumable, free-tier** — one Colab session, no paid API,
no new sweep (the expensive part is already cached).

---

## 5. Shortlist — what I'd compute first (cheapest, highest expected signal)

Ordered by (expected |ρ| ÷ cost):

1. **A4 — mean semantic-segment length** (embedding-boundary TextTiling). *Direct estimate of the
   right scale.* Highest prior.
2. **B1 — paragraph-length distribution** (mean + CV). Cheap structural twin of A4; interpretable.
3. **A2/A3 — adjacent-sentence cosine variance / topic-shift rate.** The "size-sensitivity" regime
   signal for Stage 1.
4. **C4 — gzip compressibility.** Free model-free redundancy baseline.
5. **C1 — entity density (spaCy).** "Information density" done right.
6. **E2 — gold-evidence span length** (diagnostic/oracle, to confirm the latent driver).

If #1/#2 don't clear |ρ| ≥ 0.5, the semantic-locality hypothesis is itself in doubt and we lean
toward the negative-result paper, now *much* better supported (we tried the causally-right
features, not just surface stats).

---

## 6. Risks & honest failure modes

- **Small N (50 docs, ~3.4 Q/doc).** The target is noisy; we showed even clean 0.38 wins only
  ~half. A new feature must be *strong* (≥0.5), not marginally better. More QASPER docs (cheap,
  cached/resumable) raises power — worth scaling to ~150 if a feature looks marginal.
- **Flat-curve confound (E1).** Many docs may be size-insensitive; including them dilutes any
  correlation. Stratifying on curve sharpness could *reveal* a signal that's there only for
  size-sensitive docs — but that's also a smaller, weaker claim.
- **Overfitting the combiner.** ≤3 features, monotone, held-out 50-split, report win-rate + std.
- **Generalization.** Even a QASPER win must be checked on a second corpus before any claim.
- **Novelty creep toward MoG/HiChunk.** If we go learned + query-aware we lose the "training-free"
  differentiator. Stay document-intrinsic + closed-form calibration.

---

## 7. Open decisions for next session (my recommendation in **bold**)

1. **Which features to implement first?** → **Shortlist #1–#4 (A4, B1, A2/A3, C4)** — all SBERT/CPU,
   one cheap Colab pass on the cached sweep.
2. **Deployable scope: document-intrinsic only, or allow query-aware?** → **Document-intrinsic only**
   (build-time constraint); query-aware noted as future work.
3. **Combiner: direct estimator vs non-linear?** → **Direct estimator on A4/B1 first**; escalate to
   a 2-feature monotone interaction only if single features land marginal.
4. **Scale N if marginal?** → **Yes, to ~150 docs** (resumable sweep) before any positive claim.

---

### Pointers
- Current features: `raptor/chunking/density_score.py` (the 6 surface features)
- Oracle target + harness: `experiments/granularity_sweep.py`, `experiments/granularity_calibration.py`
- The bar / power tool: `experiments/signal_recovery_power.py`,
  `docs/results/2026-06-18-signal-recovery-power.md`
- H2 negative result: `docs/results/2026-06-15-h2-density-50doc.md`
- Structure axis (boundary): `raptor/chunking/structure_score.py`
