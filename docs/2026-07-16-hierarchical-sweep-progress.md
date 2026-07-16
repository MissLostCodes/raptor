# Progress Report — Does the Negative Result Survive a REAL RAPTOR Tree?

**Date:** 2026-07-16 · **Branch:** `ckraptor` · **Commit:** `8132b10` · **Author:** Shagun Gupta

**Who this is for:** my guide, and anyone following the adaptive-chunking work. This report
covers the *Phase D* experiment run since the last briefing: we replaced the cheap "flat"
proxy with **real RAPTOR trees** (LLM-summarized hierarchies) and re-ran the whole leaf-size
sweep to check whether our earlier negative result still holds. It states what we found, what
is solid, what a reviewer would challenge, and exactly what remains to finish the paper.

---

## 1. The 30-second summary

Our earlier papers used a **flat index** as a stand-in for RAPTOR because building real trees
costs LLM calls. The obvious reviewer objection was: *"your whole result is on a proxy, not a
real tree."* This experiment answers that objection with data.

**Verdict: the negative result survives, and we found something stronger.**

1. **The adaptive-chunking headroom is real** in real trees (+13.1%), even larger than in the
   flat proxy (+9.8%). The proxy did **not** invent the headroom.
2. **New, model-independent finding:** in a real tree, *leaf size and tree depth are the same
   knob*. You cannot tune one without moving the other. The per-document "best chunk size"
   found on a flat index transfers to the real tree on **only 22% of documents**. So the
   common practice of "tune chunk size on a cheap flat index, deploy on a tree" **does not
   work**.
3. **One claim is not yet safe to make:** the real tree scored *lower* than flat overall, but
   the summarizer we used (gpt-oss-120b) misbehaved — it produced summaries *longer than the
   text they summarized*. So we cannot yet say "hierarchy hurts"; it may just be a bad
   summarizer. A **free control experiment is set up** to settle this (Section 5).

---

## 2. What we ran

- **Data:** QASPER, the paper's H1/H2 cohort (**50 documents, 171 questions**).
- **Grid:** leaf sizes `{50, 100, 150, 200, 300, 400}` tokens; retrieval budget 2000 tokens.
- **Two modes on the *same* documents and *same* metric:**
  - **FLAT** — SBERT + FAISS index (the paper's existing proxy, no LLM).
  - **TREE** — a **real RAPTOR tree** per (document, leaf size): cluster → LLM-summarize →
    collapsed-tree retrieval. Summarizer = `openai/gpt-oss-120b` via OpenRouter.
- **Metric:** mean gold-evidence coverage (retrieval quality only; the QA step is stubbed, so
  no answer-generation cost or noise). Higher = the retrieved chunks contain more of the gold
  evidence paragraphs.
- **Cost:** ~$0.09, ~4 hours on a Kaggle T4, resumable (survived an API-key expiry mid-run).

---

## 3. Results (all numbers reproduced from the committed raw records)

### 3.1 Coverage by leaf size

| leaf size | FLAT coverage | TREE coverage | TREE mean layers | TREE 0-layer docs |
|----------:|:-------------:|:-------------:|:----------------:|:-----------------:|
| 50  | 0.656 | 0.145 | 1.84 | 0 %  |
| 100 | 0.715 | 0.377 | 1.26 | 0 %  |
| 150 | 0.721 | 0.598 | 0.98 | 2 %  |
| 200 | **0.736** | 0.618 | 0.96 | 4 %  |
| 300 | 0.691 | 0.644 | 0.74 | 26 % |
| 400 | 0.713 | **0.661** | 0.52 | 48 % |

### 3.2 Headroom (per-document oracle vs best single fixed size)

| mode | best fixed size | coverage | per-doc oracle | headroom |
|------|:---------------:|:--------:|:--------------:|:--------:|
| FLAT | 200 tok | 0.736 | 0.808 | **+9.8 %** |
| TREE | 400 tok | 0.661 | 0.748 | **+13.1 %** |

### 3.3 The confound — leaf size and tree depth are inseparable

- Mean tree depth falls **1.84 → 0.52 layers** as leaf size grows 50 → 400.
- At leaf size 400, **48% of documents do not cluster at all** (0 layers) — "RAPTOR"
  silently degenerates into a flat pile of leaves.
- The flat and tree **per-document optima agree on only 11/50 documents (22%)**. Flat wants
  small chunks (optima concentrated at 50–100 tok); the tree wants large chunks (optima at
  150–400 tok). This is the core evidence that leaf size is not an independent variable in a
  hierarchical retriever.

### 3.4 Aggregation is harder (the SLIDERS axis, `m` = # gold-evidence paragraphs)

| question type | FLAT coverage | TREE coverage |
|---------------|:-------------:|:-------------:|
| `m = 1` (lookup)       | 0.848 | 0.601 |
| `m = 2` (multi-hop)    | 0.622 | 0.453 |
| `m ≥ 3` (aggregation)  | 0.675 | 0.493 |

Coverage drops as questions require more scattered evidence, in both modes — motivating the
structured-extraction direction for aggregation questions.

---

## 4. What this means for the paper

**The defensible headline (write this):** *In hierarchical retrieval, chunk granularity and
tree depth are entangled; the per-document optimum measured on a flat index transfers to a
real tree only 22% of the time, so flat-proxy chunk-size tuning does not carry over to
hierarchical RAG.* This is fully supported and model-independent (pure clustering geometry).

**Do NOT write** "RAPTOR is worse than flat retrieval." The data shows it here, but the result
is confounded by the misbehaving summarizer and would be rejected. It becomes a *secondary
observation with an explicit caveat*, pending Section 5.

**Honest venue read:** at n=50 / one corpus / one summarizer this is a **workshop or short
paper**. To reach a Findings-tier submission we need the scaling in Section 6.

---

## 5. In progress — the free de-confound (do this first tomorrow)

**Question:** is the tree worse because of the *hierarchy*, or because gpt-oss produced
bloated summaries that polluted the tree nodes?

**Control:** re-run the tree sweep with an **extractive summarizer** (`OfflineSummarizer` in
`experiments/tree_cost_dryrun.py`) — short, faithful, bounded summaries, **zero API cost**,
everything else held fixed. Two ready-to-paste Kaggle cells exist (Cell A = the sweep, Cell B
= the three-way comparison).

**Reading the outcome:**
- Extractive tree **recovers toward flat** → the summarizer was the bottleneck. Claim:
  *hierarchy is summarizer-bound, not structurally worse.*
- Extractive tree **stays low** → the hierarchy itself underperforms on evidence coverage,
  independent of summarizer. A cleaner, stronger structural claim, and gpt-oss stops being a
  liability.

Either outcome removes the biggest reviewer objection, likely **without a second paid run**.
Output file to produce: `experiments/results/sweep_tree_offline_n50.json`.

---

## 6. Remaining steps to finish the research

Ordered by priority. Items 1–2 make the result *safe*; 3–5 make it *strong enough for a real
venue*; 6 is the writeup.

- [ ] **1. De-confound (free, ~30 min).** Run the extractive-summarizer control (Section 5),
      commit `sweep_tree_offline_n50.json`, and lock in the corrected "tree vs flat" claim.
- [ ] **2. H2 on real trees.** Feed the tree records (they already carry `per_question` + `m`)
      through `notebooks/headroom_decomposition.ipynb` to test whether the tree's per-document
      optimum is any more *predictable* from density features than the flat one was. This
      extends the paper's H2 negative result to real trees.
- [ ] **3. Scale up (removes "n=50 is a pilot").** Re-run the flat + tree sweep on the **full
      QASPER test split (~416 docs)**. The re-aggregation infrastructure already exists; the
      sweep is resumable. Estimated tree cost ~$0.78.
- [ ] **4. Second corpus (removes "one dataset").** Repeat on **QuALITY** (loader already in
      `experiments/datasets.py`) to show the confound generalizes beyond QASPER.
- [ ] **5. (Optional) Downstream QA.** Swap the stub QA model for a real one on a small slice
      to confirm the retrieval-coverage gap actually moves answer quality — pre-empts the
      "so-what for the end task?" question.
- [ ] **6. Write the paper.** Structure around the entangled-knob finding (Section 4), with the
      summarizer-controlled tree comparison and the SLIDERS aggregation axis as support.

---

## 7. Reproducibility / where everything lives

- **Branch/commit:** `ckraptor` @ `8132b10` (pushed to `origin`).
- **Raw results (committed):** `experiments/results/sweep_flat_n50.json`,
  `experiments/results/sweep_tree_n50.json`. Each record carries `doc_id`, `size`, `n_chunks`,
  `n_layers` (tree depth; `null` for flat, `0` for a degenerate tree), and per-question
  coverage + evidence multiplicity `m`.
- **Experiment code:** `experiments/granularity_sweep.py` — `run_sweep` (resumable, mode-safe:
  refuses to mix flat and tree caches), `build_tree_retriever`, `sweep_document`.
- **Notebook:** `notebooks/hierarchical_sweep.ipynb` (runs on Kaggle/Colab, resumable).
- **Tests:** `tests/test_sweep_tree_mode.py` (flat/tree cache-mode safety).

---

*Status: experiment complete, verdict committed, de-confound control set up and ready to run.
Resume at Section 6, item 1.*
