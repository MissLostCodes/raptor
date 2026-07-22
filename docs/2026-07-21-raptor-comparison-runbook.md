# Runbook — RAPTOR Comparison Study (chunk-size × hierarchy × question-type)

**Date:** 2026-07-21 · **Branch:** `ckraptor` · **Author:** Shagun Gupta

**What this is:** the concrete experiment + Kaggle plan that turns the current flat-proxy
negative-result paper into a **comparison / characterization of RAPTOR-style hierarchical
retrieval**. It closes the two weaknesses the paper *already admits to* in its Threats section
(`paper/main.tex`): the flat-index proxy, and coverage-not-QA. Nothing here fabricates a result;
the "positive-shaped" findings come from measuring things RAPTOR never measured.

> **Framing (locked, see `docs/2026-07-20-problem-statement-framing.md`):** we are **not**
> re-testing "tree beats flat" (RAPTOR settled that on average, on QA). We compare the tree
> **against itself across chunk sizes** and **against flat conditioned on question type**, on the
> metric that matters (downstream QA), and we quantify how fragile the hierarchy is to a knob
> nobody tunes.

---

## 1. Directions → experiments (what's already done vs. genuinely new)

| # | Direction | Status | What produces it |
|---|-----------|--------|------------------|
| 2 | **Chunk-size sensitivity** (leaf 50→400) | **DATA EXISTS** (`sweep_tree_n50.json`) | already swept; needs de-confound + scale to confirm |
| 3 | **Hierarchy collapse** (depth vs leaf size) | **DATA EXISTS** — depth 1.84→0.52, 48% degenerate @400 | `n_layers` per cell, already recorded. Verified 2026-07-21. |
| — | **De-confound** (extractive summarizer) | **NOT RUN** — blocker for any "tree < flat" claim | Cell A below → `sweep_tree_offline_n50.json` |
| 4 | **Question-type analysis on QA** (m=1/2/≥3) | **NOT RUN** — the load-bearing new result | Cell B+C below (`qa_model` path, new code) |
| 5 | **Robustness on QA** (does the size cliff move answers?) | **NOT RUN** | falls out of Cell B (answer-F1 across sizes) |
| 6 | **Collapsed vs tree-traversal × chunk size** | **NOT RUN** — small, secondary | one-line toggle (§4), optional |

**Key point:** Directions 2 & 3 are essentially *done* (measured, model-independent geometry).
The genuinely new experiment is **downstream QA** (Directions 4 & 5). That code now exists:
`sweep_document(..., qa_model=...)` answers each question from its retrieved context and grades
it with `answer_f1` (QASPER metric). Tested in `tests/test_answer_f1.py`.

---

## 2. Run order

1. **Cell A — de-confound** (free, ~30 min). Makes the "tree vs flat" direction safe. Do first.
2. **Cell B — QA sweep** (flat + tree × sizes, ~$0.10–0.20 on top of cached summaries). The new signal.
3. **Cell C — QA analysis** (free, local). answer-F1 by mode × size × question-type.
4. **Scale** (only for whichever of the above shows signal): bump `N_DOCS` to the full split; the
   sweep is resumable, so just re-run with a bigger loader limit and a new `out_path`.
5. **Update paper** with real numbers (depth-collapse section already added; QA tables are `\todo`).

All cells **reuse setup cells 1–3 of `notebooks/hierarchical_sweep.ipynb` verbatim** (git clone,
secrets, `docs = get_loader('qasper').load(limit=N_DOCS)`, and `RUN_DIR`/`*_OUT` paths). Only the
new cells are below.

---

## 3. Ready-to-paste Kaggle cells

### Cell A — de-confound (extractive summarizer, free)
Isolates "is the tree worse because of the *hierarchy* or because gpt-oss made bloated summaries?"

```python
# A) DE-CONFOUND: same tree sweep, but summaries are a free extractive slice.
#    Tree recovers toward flat -> summarizer was the bottleneck.
#    Tree stays low          -> the hierarchy itself underperforms (cleaner claim).
from experiments.tree_cost_dryrun import OfflineSummarizer
from tqdm.auto import tqdm

OFFLINE_OUT = os.path.join(RUN_DIR, 'sweep_tree_offline_n50.json')
tree_offline = run_sweep(
    docs, SIZES, budget=BUDGET, out_path=OFFLINE_OUT,
    summarization_model=OfflineSummarizer(),      # $0, no API calls
    progress=lambda m: tqdm.write(m),
)
print(f'offline tree: {len(tree_offline)} (doc, size) records -> {OFFLINE_OUT}')
```

### Cell B — downstream QA sweep (flat + tree × sizes) — THE new experiment
```python
# B) QA SWEEP: answer each question FROM its retrieved context, grade vs gold (answer-F1).
#    Same qa_model for flat and tree -> a fair answer-quality comparison, not just coverage.
from experiments.config import ExperimentConfig
from experiments.models import build_models
from tqdm.auto import tqdm

cfg = ExperimentConfig(model=MODEL, cache_dir=os.path.join(RUN_DIR, '.llm_cache'))
summarization_model, qa_model = build_models(cfg)   # DiskCache-backed: reuses cached summaries

FLAT_QA_OUT = os.path.join(RUN_DIR, 'sweep_flat_qa_n50.json')
TREE_QA_OUT = os.path.join(RUN_DIR, 'sweep_tree_qa_n50.json')

flat_qa = run_sweep(docs, SIZES, budget=BUDGET, out_path=FLAT_QA_OUT,
                    qa_model=qa_model, progress=lambda m: tqdm.write(m))
tree_qa = run_sweep(docs, SIZES, budget=BUDGET, out_path=TREE_QA_OUT,
                    summarization_model=summarization_model, qa_model=qa_model,
                    progress=lambda m: tqdm.write(m))
print(f'flat+QA: {len(flat_qa)} | tree+QA: {len(tree_qa)} records')
```
*Cost note:* the tree run reuses cached summaries from your earlier tree sweep (same `cache_dir`),
so you pay only the QA calls: ~50 docs × ~3.4 q × 6 sizes × 2 modes ≈ 2k calls, ~$0.10–0.20 paid.
Resumable — re-run after a timeout and it skips done `(doc,size)` pairs.

### Cell C — QA analysis: answer-F1 by mode × size × question-type
```python
# C) Where does the tree actually help on ANSWERS? (Directions 4 & 5)
import json, statistics as st
from collections import defaultdict

def bucket(m):
    return '1 lookup' if m == 1 else ('2 multi-hop' if m == 2 else '>=3 aggregation')

def by_size_f1(recs):
    d = defaultdict(list)
    for r in recs:
        if r.get('mean_answer_f1') is not None:
            d[r['size']].append(r['mean_answer_f1'])
    return {s: st.mean(v) for s, v in d.items()}

def by_type_f1(recs):
    d = defaultdict(list)
    for r in recs:
        for pq in r.get('per_question', []):
            if 'answer_f1' in pq and pq.get('m'):
                d[bucket(pq['m'])].append(pq['answer_f1'])
    return {k: (st.mean(v), len(v)) for k, v in d.items()}

flat_qa = json.load(open(FLAT_QA_OUT)); tree_qa = json.load(open(TREE_QA_OUT))
print('answer-F1 by leaf size:  size   flat    tree')
fs, ts = by_size_f1(flat_qa), by_size_f1(tree_qa)
for s in sorted(fs):
    print(f'   {s:>4}   {fs[s]:.3f}   {ts.get(s, float("nan")):.3f}')

print('\nanswer-F1 by question type (best size per mode):')
for k, (v, n) in sorted(by_type_f1(tree_qa).items()):
    fv = dict(by_type_f1(flat_qa)).get(k, (float('nan'),))[0]
    print(f'   {k:<16} flat {fv:.3f}   tree {v:.3f}   (n={n})')
```

---

## 4. Direction 6 (optional, small): collapsed vs tree-traversal × chunk size

`build_tree_retriever` hardcodes `collapse_tree=True`. To add the traversal arm, thread a flag:

```python
# in experiments/granularity_sweep.py, build_tree_retriever(...):
def build_tree_retriever(doc, size, budget, summarization_model, embedding_model=None,
                         collapse_tree=True):          # <- add
    ...
    def retrieve(question):
        return ra.retrieve(question, max_tokens=budget,
                           collapse_tree=collapse_tree,  # <- use
                           return_layer_information=False)
```
Then run the QA sweep twice (collapse True/False) and compare. Do this only if the primary QA
result lands — it's a secondary "why collapsed wins, and does chunk size change that" angle.

---

## 5. What each result lets you claim (honestly)

- **Depth collapse (done):** "leaf size and tree depth are one coupled knob; at 400 tok, 48% of
  QASPER 'trees' don't cluster at all." Pure geometry, model-independent. **Safe now.**
- **De-confound (Cell A):** either "hierarchy is summarizer-bound" or "hierarchy structurally
  underperforms on coverage" — both are real, and both retire the gpt-oss objection.
- **QA by question-type (Cell B/C):** the positive-shaped finding — *when* the tree helps
  (expected: aggregation/multi-hop) vs. hurts (lookup), on answer quality. If it helps nowhere,
  that's "RAPTOR's gains are chunk-size-fragile / don't replicate here" — a qualification of a
  famous result, still publishable, still **not** a boring null.
- **Robustness (Cell B across sizes):** how far answer-F1 falls when leaf size is mis-set —
  a deployment warning nobody has quantified.

---

## 6. Open decisions

- **QA metric:** using **QASPER token answer-F1** (`answer_f1`, standard + RAPTOR-comparable).
  For QuALITY later, switch to option accuracy (multiple-choice) — separate path.
- **Paper structure:** **fold into the existing paper** (recommended) — the tree+QA work resolves
  its own stated threats, upgrading it from workshop negative result toward Findings-plausible
  characterization. A companion second paper is the alternative if it grows too large.
- **Novelty:** pending the running literature check (real citations) before finalizing Related Work.
```
