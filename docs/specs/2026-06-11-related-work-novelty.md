# Related-Work Novelty Due-Diligence Memo

**Date:** 2026-06-11 · **Status:** novelty gate for the density-adaptive leaf-granularity paper
**Scope:** positioning vs the closest prior art on adaptive chunk granularity for RAG.
**Sources read:** arXiv:2406.00456 (Mix-of-Granularity, HTML v1), arXiv:2509.11552 (HiChunk, HTML v2),
arXiv:2505.21700 (Rethinking Chunk Size, abstract), arXiv:2603.25333 / `ekimetrics/adaptive-chunking`
(Adaptive Chunking). All four arXiv IDs were **re-verified against the live web on 2026-06-11**
(titles/IDs/venues confirmed real — MoG = COLING 2025, Ekimetrics = LREC 2026). For Ekimetrics the
**repo README + the LightOn blog were read directly** (arXiv HTML 404s), which **confirms the three
claims our differentiation depends on** (details in Threat #1): Size Compliance = "fraction of chunks
within target token-count bounds" (a binary penalty, not a size policy); the split-then-merge splitter
uses **fixed per-method targets** (e.g. 600 vs 1100 tokens) and the framework **selects the best
discrete variant per document**, not a continuous size; retrieval is **flat** (no tree); and the
selection is **closed-form** (intrinsic metrics + an LLM-regex splitter, nothing trained).

---

## Our contribution (the thing under defense)

A **training-free, deterministic, two-axis chunking controller** for hierarchical (RAPTOR-style)
retrieval. Axis 1 = a closed-form **structure score** that sets chunk BOUNDARIES / routing. Axis 2 =
a closed-form **information-density score** that sets per-document / per-section **leaf SIZE**
(granularity). The density→size map is **calibrated, not learned end-to-end** — its few parameters are
fit against an **oracle leaf-size sweep** using QASPER gold-evidence coverage, on a held-out split with
no test-time tuning. Claims to defend: (1) closed-form / training-free, (2) interpretable, (3) specific
to hierarchical-tree LEAF granularity, (4) validated against a per-document oracle and shown competitive
with a learned regressor on the same features.

---

## Comparison table

| Paper (arXiv ID) | Granularity unit | Learned or closed-form | Hierarchical-leaf? | Sets size or boundary | Datasets |
|---|---|---|---|---|---|
| **Mix-of-Granularity / MoG, MoGG** (2406.00456) | **per-QUERY** (router takes query embedding) | **Learned** — supervised MLP router, BCE loss on TF-IDF/RoBERTa soft labels (offline-generated) | **No** — flat multi-granularity (coarse = concatenations of fine); MoGG = sentence-pair graph | Selects/**mixes size** (granularity weights), query-time; not boundaries | 5 medical QA: MMLU-Med, MedQA-US, MedMCQA, PubMedQA, BioASQ |
| **HiChunk + Auto-Merge** (2509.11552) | **per-QUERY** at retrieval (Auto-Merge); structure is per-document | **Learned** — fine-tuned Qwen3-4B for structure detection (trained on Gov-report, Qasper, Wiki-727) | **Yes** — multi-level tree (sections→paragraphs) | Sets **boundaries / hierarchy**; fixed-size chunking applied after (e.g. HC200); Auto-Merge merges children→parents at query time by budget | Qasper, GutenQA, OHRBench, LongBench, HiCBench |
| **Rethinking Chunk Size** (2505.21700) | **per-COLLECTION** (one fixed size, dataset-dependent) | **Closed-form / empirical** (no model trained; a benchmark study) | **No** — flat fixed-size | **Size** — but global, chosen empirically per dataset, no per-doc policy | short- and long-form retrieval sets (Stella/Snowflake embedders) |
| **Adaptive Chunking** (2603.25333, Ekimetrics) | **per-DOCUMENT** | **Closed-form** intrinsic metrics (RC, ICC, DCC, BI, SC) select a chunking **method**; no trained selector model | **No** — flat; includes a split-then-merge recursive splitter, but not a RAPTOR-style retrieval tree | Selects a **method** per doc (not a continuous size); SC = size-compliance is a constraint, not a size policy | legal / technical / social-science corpus (LREC 2026) |
| **OURS** (this work) | **per-DOCUMENT / per-SECTION** | **Closed-form / training-free** — calibrated density→size map (few params, oracle-fit, held-out) | **Yes** — sets RAPTOR-tree LEAF granularity | Sets **both**: structure score → boundaries, density score → **leaf size** (two decoupled axes) | QASPER (gold-evidence calibration + QA), QuALITY, NarrativeQA |

---

## What is UNIQUE to us (the defensible niche)

The defensible cell is the **conjunction**, and no single competitor occupies it: a method that is
simultaneously (a) **closed-form / training-free**, (b) sets a **continuous per-document/per-section
LEAF SIZE**, and (c) does so on a **hierarchical RAPTOR-style tree**, (d) **calibrated against a
per-document oracle** rather than hand-tuned. Each prior work misses at least one:

- **MoG** is *learned* and *per-query* and *flat* — it misses (a), (b/c granularity-unit), and the tree.
- **HiChunk** is *learned* (fine-tuned LLM) and sets *boundaries* (size is a fixed post-step), and its
  adaptivity (Auto-Merge) is *query-time*, not a per-document size policy — misses (a) and "sets size".
- **Rethinking Chunk Size** is closed-form but *global per-collection* and *flat* — misses (b) per-doc
  and (c) tree.
- **Adaptive Chunking (Ekimetrics)** is the closest on the closed-form + per-document axes, but it
  selects a discrete **method**, not a continuous **size**, it is **flat (not a retrieval tree)**, and
  its metrics are used as selection heuristics, **not calibrated against an oracle size sweep**.

Also genuinely novel and worth foregrounding: the **two-axis decomposition itself** — separating
"where to cut" (structure) from "how big to make leaves" (density) as two cheap deterministic scores.
No read paper states this decomposition cleanly; competitors collapse granularity into one learned
selector (MoG) or one structural detector (HiChunk).

### Claims AT RISK (a reviewer could say "already done")

- **"Adaptive chunk size to content" as a broad idea is NOT novel.** MoG, the Ekimetrics work, and the
  2505.21700 study all argue size should adapt. Do not claim the *concept*; claim the *specific
  mechanism and setting*.
- **"Per-document, closed-form, feature-driven chunk selection"** is partially occupied by Ekimetrics
  (2603.25333). This is the single biggest collision. Our differentiation must be sharp: continuous
  **size** vs their discrete **method**; **RAPTOR leaf** vs their flat chunks; **oracle-calibrated** vs
  their intrinsic-heuristic selection. If a reviewer reads SC (Size Compliance) loosely, they may claim
  size is already handled — pre-empt this by stating SC is a *constraint check*, not a size *policy*.
- **"Hierarchical + adaptive granularity"** is occupied by HiChunk — but theirs is a *fine-tuned LLM*
  doing *boundaries* with *query-time* merging. Our defense is training-free + per-document size, not
  query-time. Keep that distinction explicit or the hierarchical claim weakens.
- **"Competitive with a learned selector"** is a claim, not yet a result (H2/H3 pending). It is only
  defensible once E2/E3 land; until then it is an aspiration, not novelty.
- **"Training-free" is NOT a differentiator versus Ekimetrics** (verified: their selection is also
  closed-form, nothing trained). It is a clean differentiator only versus the *learned* MoG/HiChunk.
  Against Ekimetrics, lead with **continuous size vs discrete variant**, **hierarchical tree vs flat**,
  and **oracle calibration vs intrinsic-metric heuristic** — never with "we don't train a model," which
  they can also say.

---

## Positioning language for the paper's Related Work (paste-ready)

> Mix-of-Granularity (MoG; arXiv:2406.00456) is the closest prior work on adaptive retrieval
> granularity, but it differs from ours on three axes. First, MoG is **learned**: it trains a
> supervised MLP router (binary-cross-entropy over TF-IDF/RoBERTa soft labels) to weight granularity
> levels, whereas our controller is **closed-form and training-free**, with a handful of parameters
> calibrated once against a per-document oracle. Second, MoG selects granularity **per query** at
> retrieval time over a **flat** set of multi-granularity chunks (coarse chunks are concatenations of
> fine ones); we instead set **per-document/per-section leaf size within a hierarchical RAPTOR tree**,
> as a property of the index rather than the query. Third, MoG's adaptivity is opaque (a learned weight
> vector), while our density→size map is **interpretable**: one can read off why a given section
> received its granularity. Relative to HiChunk (arXiv:2509.11552), which fine-tunes an LLM to detect
> structure and merges chunks at query time, we require **no model training** and adapt **size**, not
> just boundaries; relative to method-selection approaches (Ekimetrics, arXiv:2603.25333) we choose a
> **continuous leaf size** rather than a discrete chunking method, on a retrieval tree rather than flat
> chunks. To our knowledge, no prior work decomposes chunking control into two cheap deterministic
> axes — structure→boundary and density→size — for hierarchical-tree leaf granularity.

---

## Threats to novelty (stated honestly)

1. **Ekimetrics Adaptive Chunking (2603.25333, LREC 2026) is the closest prior work — but the
   collision is NARROWER than feared (VERIFIED 2026-06-11).** It is closed-form, per-document, and
   feature-driven — three of our four pillars. The two specific worries are now **resolved against the
   repo + LightOn blog**:
   - *Does Size Compliance constitute a size policy?* **No.** SC is "the fraction of chunks within
     target token-count bounds" — a binary compliance **penalty** on a *preset* bound, not a mechanism
     that *chooses* the bound. We choose the per-document size from density; they only check conformance
     to a fixed window.
   - *Is the recursive splitter effectively a leaf-size controller?* **No.** It runs with **fixed
     per-method targets** (e.g. 600 vs 1100 tokens) and the framework **selects the best discrete
     variant per document** via the five metrics — discrete method/variant selection, not a continuous
     calibrated size. Their own text: "the splitter itself doesn't adaptively choose its own chunk
     size."
   - Retrieval is **flat** (no RAPTOR/tree/parent-summary), and nothing is trained.
   So our separation holds on **continuous size** (not discrete variant), **hierarchical RAPTOR leaf**
   (not flat), and **oracle calibration** (not intrinsic-metric heuristic). A full PDF read before
   camera-ready is still worthwhile, but the differentiation no longer rests on an unread claim.
2. **The broad idea "adapt size to information density" is folklore + published.** Practitioner
   heuristics ("smaller chunks for dense text") plus 2505.21700's empirical finding mean we cannot
   claim the concept, only the formalized, validated, hierarchical-leaf mechanism. Reviewers may judge
   the delta incremental if framed as "density-aware chunking" rather than "two-axis closed-form
   controller calibrated to an oracle."
3. **The "training-free ≈ learned" headline is unproven.** It is the crux of the contribution but is
   currently a hypothesis (H2/H3). If the closed-form map underperforms a learned regressor by a wide
   margin, the "competitive without training" claim — a key differentiator from MoG — collapses.
4. **Hierarchical-leaf specificity may be a small surface in practice.** If the leaf-size sweep (H1)
   shows a flat coverage curve or one shared global optimum on QASPER, the per-document adaptivity has
   no room to operate and the contribution reduces to "pick a good fixed size," which 2505.21700 already
   covers. H1 is the genuine gate.
5. **Calibration-vs-learning is a semantic line a reviewer may push on.** "Calibrated against an oracle
   sweep" is a fitting procedure with parameters and a train/test split; an adversarial reviewer can
   call this "a (small) learned model," eroding the clean "training-free" claim. Pre-empt by stressing
   the closed-form functional form, tiny parameter count, interpretability, and zero per-query/per-doc
   inference cost.

---

## Executive summary

1. **The niche is real but narrow.** Our defensible cell is the conjunction {closed-form/training-free
   + continuous per-document leaf SIZE + RAPTOR hierarchical tree + oracle-calibrated}; no single prior
   work occupies all four, and the two-axis structure/density decomposition is itself unclaimed.
2. **Closest threat is NOT Mix-of-Granularity — it is Ekimetrics Adaptive Chunking (2603.25333),**
   which already does closed-form, per-document, feature-driven selection; our separation from it rides
   entirely on continuous-size-vs-method, tree-vs-flat, and oracle-calibration. **Verified 2026-06-11:**
   SC is a within-bounds compliance penalty (not a size policy), their splitter selects among fixed
   discrete variants (600/1100 tok), retrieval is flat, nothing is trained — so all three separators
   hold. Corollary: **do not use "training-free" to separate from Ekimetrics** (they are too); reserve
   it for the contrast with the *learned* MoG/HiChunk.
3. **Mix-of-Granularity (2406.00456) is cleanly differentiated:** it is learned (supervised MLP router),
   per-query, and flat — we are training-free, per-document, and hierarchical. This contrast is the
   strongest, most quotable part of our positioning.
4. **Do not claim the broad concept** ("adapt size to density"); it is folklore plus 2505.21700. Claim
   the specific mechanism, the hierarchical-leaf setting, and the oracle validation.
5. **Verdict: niche is clear-but-contingent.** It holds *iff* H1 (size matters and is doc-heterogeneous,
   ✅ passed on the 8-doc pilot) and H2 (closed-form selected-feature map ≈ learned regressor) land
   empirically; until then the strongest claims are aspirational. The novelty-vs-prior-art question is
   no longer a risk (the closest competitor is read and the separators are verified); the remaining risk
   is purely **empirical** (does H2 hold at ~50 docs?). A full PDF read of 2603.25333 before camera-ready
   remains nice-to-have, not blocking.
