# Plan — Weighted Adaptive Leaf-Size Chunking (executing the document-features brainstorm)

**Date:** 2026-06-27 · **Branch:** `ckraptor` · **Status:** EXECUTABLE PLAN (approved direction)
**Author:** Shagun + research agent
**Supersedes brainstorm:** `docs/specs/2026-06-18-document-features-brainstorm.md` (this turns that
brainstorm into a concrete, sequenced, pre-registered execution plan).

---

## 0. TL;DR — what this document is

This is the complete, ordered list of **everything we are going to do next, why, and how**, to
build a **weighted, training-free, per-document leaf-size predictor** ("weighted chunking") for
RAPTOR — and to either (a) capture the proven oracle headroom with it, or (b) convert the attempt
into a much stronger negative result. Every step reuses the **no-LLM cached harness** we already
built, so the feature-search loop costs effectively **zero LLM budget**. The only LLM cost is the
*final* end-to-end QA confirmation (Phase 5), and only if the proxy results earn it.

**North star:** capture the **+9.8% relative** gold-evidence oracle headroom (H1) with a
document-intrinsic, closed-form/monotone **weighted** size predictor — or refute that route too,
with the *causally-right* features in hand.

---

## 1. Where we are (the constraints this plan must respect)

Established, with receipts:

1. **H1 — PASS (n=50).** Leaf size matters and the optimum is heterogeneous. Best fixed size =
   200 tok @ coverage 0.7362; per-doc oracle = 0.8080 → **+9.8% rel.** headroom. Optimal-size mass
   exists at *every* grid size. *(`docs/results/2026-06-15-h2-density-50doc.md`)*
2. **H2 — FAIL (n=50).** The 6 cheap **surface** density features do **not** predict optimal size
   well enough to beat fixed=200. Best single feature `type_token_ratio` ρ=−0.379; selected-feature
   map wins only **7/50** splits (14%). *(same source)*
3. **A weighted linear combo is ALREADY refuted.** The "learned 6-feature regressor" is exactly an
   optimally-weighted linear sum of those features and it *also* lost (0.707 < 0.729). **⇒ Re-weighting
   the old 6 features cannot win.** Any new weighted strategy MUST bring **new features** and/or a
   **non-linear / staged** combiner.
4. **The bar (power analysis).** A clean signal needs **|ρ| ≳ 0.6** to reliably (≥80% of splits)
   beat fixed; even a *clean* 0.38 wins only ~50%, and QASPER's real 0.38 won 14%. **Target for any
   new feature: |ρ| ≳ 0.5–0.6 held-out.** *(`docs/results/2026-06-18-signal-recovery-power.md`)*
5. **Deployability envelope.** RAPTOR fixes leaf size at *tree-build time, per document, before any
   query exists.* So every shippable feature must be computable from the **document alone**, cheaply:
   SBERT is already in the pipeline; spaCy / gzip are cheap; **no paid APIs, free-tier T4**.

**The reframe driving the new features (H-new):** optimal leaf size tracks a document's intrinsic
**semantic segment size / evidence locality** — *not* surface lexical density. Short, self-contained
evidence spans → small leaves win; evidence spread across a coherent passage → large leaves win.
Surface features (TTR, word length) are only loose proxies for this; semantic-segment features
measure it directly.

---

## 2. The end product we are building: the "weighted chunking" model

The honest instantiation of "weighted chunking" (from brainstorm §3, recommended option):
a **2-stage, ≤3-feature, monotone** per-document size predictor.

```
predict_leaf_size(doc):
    # Stage 1 — REGIME GATE: is this document size-sensitive at all?
    if regime_signal(doc) < τ:          # flat oracle curve → size barely matters
        return FIXED_DEFAULT            # = 200 tok, the strong tuned baseline
    # Stage 2 — SIZE ESTIMATE from the causally-right scale feature(s)
    raw = a + b * natural_segment_size(doc)   # A4 (semantic seg len) or B1 (paragraph len)
    return clip(raw, 50, 400)           # smooth calibrated map, NOT a discrete jump
```

- **Stage 1 (regime)** uses a size-sensitivity signal (adjacent-sentence cosine variance A2 / segment
  CV A5). Documents with a flat oracle coverage curve are *confounds* — predicting their size is
  pointless and dilutes correlation. Gate them to the safe fixed default.
- **Stage 2 (size)** maps the natural-segment-size feature (A4 / B1) to a leaf size via a tiny,
  interpretable, monotone calibration (slope `b`, intercept `a`, clip) fit on the TRAIN split only.
- **"Weights"** = the small set of calibration parameters `{τ, a, b}` (+ at most one interaction
  term). Interpretable and hard to overfit at N=50 — this is deliberate, given constraint §1.3.

This replaces the never-injected density-arm coefficients. It plugs into the existing
`DensityAdaptiveChunker` via a calibrated size-fn (the `make_calibrated_size_fn` hook already exists).

---

## 3. The phased plan (goal · why · how · pre-registered gate · paper impact)

> **Sequencing rule:** Phases 0–3 are pure proxy / no-LLM and run in one cheap Colab session.
> Phase 5 (LLM) runs ONLY if Phase 3 clears the bar. Phase 6 (paper) runs regardless of outcome.

### Phase 0 — Target diagnostics (E2/E3): what does optimal size actually respond to?
- **Goal:** Before guessing features, measure what the optimal-size target correlates with using
  *label-derived* (near-oracle) signals, and quantify the flat-curve confound.
- **Why:** If even label-derived gold-evidence geometry can't explain optimal size, no cheap document
  feature will — that tells us the ceiling on the whole feature search up front. It also gives the
  Stage-1 regime gate its justification (how many docs are size-insensitive?).
- **How:**
  - **E1 — curve shape:** per-doc oracle coverage curve over sizes {50…400}; record peak location,
    peak sharpness, flatness. Flag "flat" docs (Δ between best and worst size < ε).
  - **E2 — gold-evidence span length:** tokens per gold-evidence paragraph, per doc; regress optimal
    size on E2 (Spearman + held-out). **Hypothesized true latent driver.**
  - **E3 — evidence dispersion:** are a question's gold spans contiguous or scattered? Correlate with
    optimal size.
  - Implement in `experiments/granularity_calibration.py` (reuse `rank_corr`, the 50-split harness);
    inputs are the **already-cached** sweep + QASPER gold evidence. No new sweep.
- **Soft gate (decision, not a hard stop):**
  - If E2/E3 explain optimal size well (|ρ|≳0.5) → strong prior that a cheap feature predicting E2
    can win; **prioritize the feature most correlated with E2** in Phase 1.
  - If E2/E3 are weak AND the Phase-1 features also come back weak → that is the honest stop signal
    (Phase 6 negative-result path). **Phase 0 alone never cancels the feature loop** — we still run
    Phases 1–3 because the flat-curve confound (E1) can mask real signal in size-sensitive docs.
- **Paper impact:** Either a mechanism for *why* features can/can't work, or the strongest possible
  "the target itself is irreducibly noisy at this scale" statement.

### Phase 1 — Compute the shortlist of causally-right features
- **Goal:** Add the evidence-locality features to the per-doc feature table.
- **Why:** These measure semantic segment size / topic shift directly (H-new), unlike the refuted
  surface features. A4/B1 are *definitionally* the right scale for a leaf.
- **How:** New module `raptor/chunking/segment_features.py` returning a `feats` dict shaped exactly
  like `density_score`'s output, so it drops into the existing calibration harness. Shortlist, in
  priority order (expected |ρ| ÷ cost):
  1. **A4 — mean semantic-segment length** (embedding-boundary / TextTiling on the sentences the
     sweep already SBERT-embeds). *Highest prior — direct estimate of the natural chunk size.*
  2. **B1 — paragraph-length distribution** (mean + CV of tokens/paragraph). Cheap structural twin
     of A4; interpretable; reuses `structure_score` ingredients.
  3. **A2 / A3 — adjacent-sentence cosine variance / topic-shift rate.** The Stage-1 regime signal.
  4. **C4 — gzip compressibility.** Free, model-free redundancy baseline.
  5. **C1 — entity density (spaCy)** — "information density done right"; only if 1–4 are marginal.
- **Cost:** SBERT vectors already computed in the sweep → A-tier is ~free; gzip free; spaCy cheap.
- **Gate:** none (computation step). Unit-test each feature (deterministic, edge cases) per repo
  convention (271 tests passing — keep that green).

### Phase 2 — Single-feature gate (the pre-registered screen)
- **Goal:** Decide which new features, if any, carry exploitable signal.
- **Why:** Cheaper to kill features one at a time before building the combiner; avoids the
  multiple-comparisons / overfitting trap that sank the n=8 preview.
- **How (per feature, all on cached sweep):**
  1. **Spearman |ρ|** vs per-doc optimal size (`gc.rank_corr`).
  2. **Held-out 50-split eval** (`select_and_fit` → `coverage_under_sizes`): does its calibrated
     size-map beat fixed=200, and in how many of 50 splits?
  3. **Power check:** compare its win-rate against the clean-signal curve in
     `experiments/signal_recovery_power.py` (is it behaving like a real signal of its measured ρ?).
- **Pre-registered criteria (set BEFORE looking — copied from brainstorm §4):**
  - ✅ **Promising:** a single new feature reaches **|ρ| ≥ 0.5 held-out** AND beats fixed in
    **≥ 30/50 splits** → proceed to Phase 3 combiner.
  - ⚠️ **Marginal:** **0.4 ≤ |ρ| < 0.5** → only pursue via the Phase-3 2-feature interaction, and
    plan to scale N (Phase 4).
  - ❌ **Dead:** all new features < 0.4 AND the Phase-0 E2/E3 diagnostics were weak → **stop**, go to
    Phase 6 negative-result path (now much stronger: we tried the causally-right features).
- **Paper impact:** the headline table — "surface features (old) vs locality features (new) vs oracle."

### Phase 3 — Build & evaluate the weighted combiner
- **Goal:** Build the §2 model and test it held-out.
- **Why:** This is the actual "weighted chunking" deliverable the user asked for.
- **How:**
  1. **Direct estimator first** (least overfit): `leaf_size ≈ a + b·A4` (or B1), 1-param global
     calibration on TRAIN, clip [50,400]. Evaluate on the 50-split protocol.
  2. **Add Stage-1 regime gate** (A2/A5): fall back to fixed=200 on flat-curve docs. Re-evaluate.
  3. **Escalate to a 2-feature monotone interaction** (e.g. `segment_size × coherence_variance`,
     or a depth-2 gradient-boosted stump) **only if** single features were marginal. Keep **≤3
     features**, monotone where possible, report win-rate + std over 50 splits.
- **Pre-registered success gate (the real H-new test):** the weighted model beats fixed=200 in
  **≥ 30/50 held-out splits** with mean TEST coverage above fixed by a margin exceeding 1 std, and
  approaches the oracle ceiling. (Mirrors the original H2 gate, applied to the new model.)
- **Paper impact:** the central result — positive (we capture the headroom) or negative (even the
  right features + a staged weighted model fall short).

### Phase 4 — Scale to ~150 docs (conditional)
- **Trigger:** any feature/combiner lands **marginal** (0.4 ≤ |ρ| < 0.5) in Phase 2/3.
- **Why:** power analysis says N=50 (~3.4 Q/doc) is a real limit; more docs raises power and could
  push a marginal feature over the bar (or confirm it stays under).
- **How:** the sweep is **resumable and cached** — extend the QASPER subset to ~150 docs, re-run the
  no-LLM sweep, re-run Phases 2–3. Still no LLM cost. Update pinned subset JSON for reproducibility.
- **Gate:** re-apply Phase-2/3 criteria at n≈150. If still under bar → negative result, but now at
  scale (strong).

### Phase 5 — Wire the winner into RAPTOR + end-to-end QA confirmation (conditional)
- **Trigger:** Phase 3 (or Phase 4) clears the success gate on the proxy metric.
- **Why:** the proxy (gold-evidence coverage) is a within-pipeline signal; a publishable *positive*
  claim needs an end-task QA win inside the frozen RAPTOR pipeline.
- **How:**
  - Inject the calibrated weighted size-fn into `DensityAdaptiveChunker` via `make_calibrated_size_fn`
    (the `(feature, a, b)` + regime params); expose as a `get_chunker("weighted", …)` arm.
  - Run the end-to-end arm via `python -m experiments.run` against fixed=200 baseline on QASPER
    (Answer token-F1, evidence-F1), free-tier (`gpt-oss-120b` + SBERT), paired bootstrap significance.
  - **Only the chunker changes** — identical embeddings/clustering/summarizer/QA/retrieval across arms.
- **Gate:** statistically significant (or clearly non-inferior) end-to-end win → positive paper.
- **Paper impact:** turns a proxy win into a leaderboard-credible claim.

### Phase 6 — Second corpus + paper finalization (always runs)
- **Goal:** Generality check + write the paper for whichever story holds.
- **Why:** a single-corpus result (QASPER) is the main external-validity threat, for *either* outcome.
- **How:**
  - Add a 2nd corpus (NarrativeQA and/or QuALITY) through the same harness; metrics already wired
    (ROUGE-L/BLEU/METEOR for NarrativeQA via `experiments/metrics/generation.py`; accuracy for
    QuALITY). Re-run the relevant phases at pilot scale.
  - Update `paper/main.tex`:
    - **If positive:** lead with the weighted method + two-axis decomposition; H1 headroom → H-new
      capture → end-to-end win → generality.
    - **If negative:** lead with the harness + H1 headroom + the *stronger* refutation ("we tested the
      causally-right semantic-locality features and a staged weighted model; the cheap training-free
      route still does not beat a tuned constant"), with E2/E3 diagnostics explaining why.
- **Paper impact:** submission-ready draft.

---

## 4. What gets built / changed (file-level map)

| Area | File | Change |
|---|---|---|
| New features | `raptor/chunking/segment_features.py` (new) | A4, B1, A2/A3, A5, C4, (C1) feature fns → `feats` dict |
| Diagnostics | `experiments/granularity_calibration.py` | add E1 curve-shape, E2 span-length, E3 dispersion regressions |
| Combiner | `raptor/chunking/density_adaptive_chunker.py` | 2-stage weighted size-fn (regime gate + calibrated map) |
| Pipeline arm | `raptor/chunking/factory.py` | `get_chunker("weighted", …)` dispatch |
| Power/eval | `experiments/signal_recovery_power.py` | reuse as-is for power checks (no change expected) |
| Tests | `tests/` | unit tests for each new feature + combiner (keep suite green) |
| Subsets | `experiments/datasets/subsets/*.json` | extend to ~150 docs if Phase 4 triggers |
| Results | `docs/results/2026-06-27-*.md` (new) | one writeup per phase, traceable numbers |
| Paper | `paper/main.tex` | Phase 6 update |

**Reused as-is (the durable harness):** `experiments/granularity_sweep.py` (cached oracle target),
the 50-split robustness protocol, paired-bootstrap significance, pinned subsets, free-tier models.

---

## 5. Pre-registered decision tree (so we can't move goalposts later)

```
Phase 0 (E2/E3) ─┐
                 ├─> Phase 1 (compute features) ─> Phase 2 (single-feature gate)
                 │
   Phase 2 result:
     ✅ |ρ|≥0.5 & ≥30/50  ─────────────> Phase 3 combiner ─┐
     ⚠️ 0.4–0.5           ─> Phase 4 (scale) ─> Phase 3 ───┤
     ❌ all <0.4 & E2/E3 weak ───────────────────────────> Phase 6 (NEGATIVE, strong)
                                                            │
   Phase 3 result:
     ✅ weighted beats fixed ≥30/50 & ~oracle ─> Phase 5 (LLM end-to-end) ─> Phase 6 (POSITIVE)
     ❌ still loses ──────────────────────────────────────> Phase 6 (NEGATIVE, strongest)
```

**Either terminal state is a paper.** A negative result here is *more* valuable than the H2 one,
because it refutes the heuristic with the features most likely to work, not just surface stats.

---

## 6. Risks & how we control them (carried from brainstorm §6)

- **Small N / noisy target.** Held-out 50-split + report win-rate & std always; scale to ~150 if
  marginal (Phase 4). Never claim a win from a single split or from train-set ρ.
- **Flat-curve confound (E1).** Quantified in Phase 0; handled by the Stage-1 regime gate, not hidden.
- **Overfitting the combiner.** ≤3 features, monotone, calibration fit on TRAIN only, evaluate on
  TEST. No free-form deep models at N=50.
- **Generalization.** No positive claim ships without Phase 6 second-corpus check.
- **Novelty creep toward MoG/HiChunk.** Stay **document-intrinsic + closed-form/monotone calibration,
  no query-awareness, no training a model** — that is our differentiator. Query-aware = future work.
- **Budget.** Phases 0–4 are no-LLM (cached). Only Phase 5 spends LLM, and only after the proxy earns it.

---

## 7. Definition of done

- [ ] `segment_features.py` implemented + unit-tested; full suite green.
- [ ] Phase 0 diagnostics written up (`docs/results/2026-06-27-target-diagnostics.md`).
- [ ] Phase 2 single-feature gate table written up (old vs new vs oracle).
- [ ] Weighted combiner implemented, evaluated held-out, written up.
- [ ] Decision-tree terminal reached and recorded (positive → Phase 5 done; negative → documented).
- [ ] `paper/main.tex` updated to the holding story; second-corpus check included.
- [ ] All numbers traceable to a committed `docs/results/*.md`, per project convention.

---

### Pointers
- Brainstorm this executes: `docs/specs/2026-06-18-document-features-brainstorm.md`
- Spec + original gates: `docs/specs/2026-06-11-density-adaptive-chunking.md`
- H1 PASS / H2 FAIL numbers: `docs/results/2026-06-15-h2-density-50doc.md`
- The bar / power tool: `docs/results/2026-06-18-signal-recovery-power.md`, `experiments/signal_recovery_power.py`
- Oracle target + harness: `experiments/granularity_sweep.py`, `experiments/granularity_calibration.py`
- Combiner hook: `raptor/chunking/density_adaptive_chunker.py` (`make_calibrated_size_fn`), `factory.py`
- Current (refuted) features: `raptor/chunking/density_score.py`
- Briefing: `docs/2026-06-18-progress-briefing.md`
