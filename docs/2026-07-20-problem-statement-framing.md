# Problem-Statement Framing — Comparison, Not a Failed Hypothesis

**Date:** 2026-07-20 · **Branch:** `ckraptor` · **Author:** Shagun Gupta

**Purpose:** Lock in *how we present the problem* so the results read as a **comparison /
characterization**, not a "negative result." Plan: **finish all experiments first, then write
the paper using this framing.** Nothing here fabricates a result — the study *is* a flat-vs-tree
comparison; we're just pointing the framing at the comparison axis instead of a pass/fail hypothesis.

---

## The core reframe

Do **not** frame around a hypothesis that can "fail." Frame around a **comparative question that
is interesting either way.**

| ❌ Hypothesis framing | ✅ Comparison framing |
|---|---|
| "Does adaptive / hierarchical chunking *improve* retrieval?" → answer "no" = failure | "How does the optimal chunking regime *differ* between flat and hierarchical retrieval, and does tuning *transfer* between them?" → answer = a characterization |

Every possible outcome becomes a comparison: tree wins somewhere → comparison; tree loses →
comparison; tuning doesn't transfer → comparison. **There is no failure branch.**

---

## Problem statement — the wording we'll use

**Option A (chosen — entangled-knob version):**

> We study how chunk granularity should be tuned for hierarchical (RAPTOR) retrieval, and whether
> tuning done on a cheap flat index transfers to a real tree. Comparing both regimes on identical
> documents, we find they optimize in **opposite directions** — flat retrieval favors small chunks,
> hierarchical retrieval favors large ones — and the per-document flat optimum transfers to the tree
> on only **22%** of documents, because in a hierarchy **leaf size and tree depth are a single coupled
> knob**.

**Option B (fallback — broader regime characterization):**

> Rather than asking whether hierarchical retrieval beats flat retrieval, we characterize **where and
> why** their optimal operating points diverge as a function of chunk granularity, tree depth, and
> evidence multiplicity.

---

## Word-swap dictionary (use in the actual writing)

| Negative voice ❌ | Comparison voice ✅ |
|---|---|
| "adaptive chunking failed to improve" | "the optimal granularity is **regime-dependent**" |
| "hierarchy is worse" | "the two regimes **trade off differently** across question types" |
| "tuning didn't help" | "flat-tuned settings **transfer at 22%** to trees" |
| "no significant gain" | "the gain is **conditional on** tree depth / evidence spread" |
| "our hypothesis was rejected" | "we **quantify the coupling** between leaf size and depth" |

---

## Honesty guardrails (do not skip)

1. **Answer "so what do I do?"** A comparison paper must end constructive, not "everything is bad."
   Our data already answers it: *tune chunk size on the tree directly, not on a flat proxy; expect
   larger leaves in a hierarchy.* Keep that sentence in — it's what turns a null into actionable guidance.

2. **Don't hard-code the direction of the flat-vs-tree gap until the de-confound runs.**
   - SAFE regardless of de-confound (pure clustering geometry): "opposite optima", the 22% transfer,
     depth collapse 1.84→0.52, 48% degenerate trees at 400 tok. → Option A is safe now.
   - NOT yet safe: "the tree is *worse* than flat" — currently confounded by the gpt-oss summarizer
     that produced summaries longer than their source. Settle with the extractive-summarizer control
     (`sweep_tree_offline_n50.json`) before writing that claim either way.
   - The comparison frame survives **both** de-confound outcomes (summarizer-bound OR structurally
     worse) — so we can commit to the framing now, just not to the gap's sign.

---

## Evidence already in hand (n=50 QASPER, 171 Q — from the committed raw records)

- FLAT best coverage 0.736 @200 tok; TREE best 0.661 @400 tok.
- Per-document optima agree flat↔tree on only **11/50 docs (22%)**.
- Mean tree depth falls **1.84 → 0.52 layers** as leaf size grows 50 → 400.
- **48%** of docs don't cluster at all (0-layer "tree") at 400 tok.
- Adaptive headroom is **larger** in trees: **+13.1%** (tree) vs +9.8% (flat).
- Coverage falls with evidence multiplicity `m` in both modes (SLIDERS aggregation axis).

---

## Status / next

- **Direction:** locked — sell the entangled-knob comparison honestly (workshop / short-paper scope).
- **Blocking experiment:** de-confound (`sweep_tree_offline_n50.json`) — free, ~30 min, not yet run.
- **Then:** H2 on tree records → (optional) scale-up → write paper around Option A.
