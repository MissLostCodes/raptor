# Progress Report — Adaptive Leaf Granularity for Hierarchical Retrieval

**Date:** 2026-07-02 · **Branch:** `ckraptor` · **Author:** Shagun Gupta

**Who this is for:** anyone with **zero background** on this project. This report explains
what problem we are working on, why it matters, what we built, what we found, and what the
newest step (a second-corpus check + the paper draft) adds. No prior knowledge of the code,
the papers, or the jargon is assumed. Technical receipts are linked at the end for people who
want to dig in.

---

## 1. The 30-second summary

We tested a popular but never-properly-checked rule of thumb used in AI search systems —
*"break dense text into smaller pieces"* — and found, carefully and repeatedly, that **it
does not actually help**. A single, well-chosen fixed piece size works just as well or better.
We proved this on two very different collections of documents, ruled out the obvious "maybe
our tool is broken" explanations, and wrote it up as an **honest negative result** plus a
**reusable, cheap testing tool** that let us reach that conclusion without expensive AI calls.

That's the whole story. The rest of this document unpacks every term in that paragraph.

---

## 2. Background — the concepts, in plain English

### 2.1 What is retrieval-augmented generation (RAG)?

When you ask an AI assistant a question about a specific document (say, a research paper or a
novel), the AI does not read the whole thing every time. Instead the system:

1. **Chops the document into pieces** ("chunks").
2. **Stores** those pieces in a searchable index.
3. When you ask a question, it **retrieves** the few pieces most relevant to your question.
4. It hands only those pieces to the AI to write the answer.

This "look up the relevant bits, then answer" pattern is called **retrieval-augmented
generation (RAG)**. The quality of the final answer depends heavily on step 1 and step 3: if
the retrieved pieces don't contain the actual evidence, the AI can't answer well.

### 2.2 What is RAPTOR (the "hierarchical" part)?

**RAPTOR** is a well-known RAG method. Instead of a flat pile of chunks, it builds a **tree**:
it groups similar small chunks together, uses an AI to write a short **summary** of each group,
then groups and summarizes the summaries, and so on. The result is a tree where the bottom
("leaves") are the raw small chunks and higher levels are increasingly big-picture summaries.
Searching this tree can find both fine details and broad themes.

The smallest pieces at the bottom of the tree are called **leaf chunks**. RAPTOR just cuts
them to a **fixed size** — roughly 100 words each — the **same size for every document**,
whether it's a terse math paper or a rambling short story.

### 2.3 The lever we are studying: "leaf granularity"

**Leaf granularity = how big each leaf chunk is.** This is the one knob we turn. Everything
else in the pipeline (the AI models, the grouping, the search settings) is **frozen** so that
any difference in results is caused *only* by the chunk size. That makes it a clean scientific
experiment.

### 2.4 The rule of thumb we are testing

A very common piece of folklore in this field says:

> *"Denser, information-heavy text should be cut into smaller chunks; simpler text can use
> bigger chunks."*

It sounds reasonable, and lots of practitioners believe it. But — surprisingly — **nobody had
cleanly tested whether it's true for the leaf chunks of a hierarchical system like RAPTOR.**
That is exactly the gap this project fills.

### 2.5 The specific research question

> Can we pick the best chunk size **per document**, using a **cheap** calculation, **without
> training a machine-learning model** — and does that beat just picking one good fixed size for
> everything?

Three conditions matter here and are the whole point:
- **Cheap** — no expensive AI calls; just simple text statistics.
- **Training-free** — no model that has to learn from labeled data.
- **Per-document** — a size chosen individually for each document from its own text.

---

## 3. What we built: a cheap testing tool ("the harness")

Normally, to know "which chunk size is best for this document," you'd have to run the full
expensive AI pipeline for every document at every candidate size and grade the answers. That
costs a fortune in AI calls.

Our key infrastructure contribution is a **training-free, no-AI evaluation harness** that
sidesteps that cost. Here's how it works:

- For each document, and for each candidate leaf size (**50, 100, 150, 200, 300, 400 words**),
  it cuts the document, indexes the pieces, and retrieves the relevant ones for each question —
  **using only a lightweight text-embedding model, never a full language model.**
- Instead of grading the final written answer, it uses a cheap **proxy score**: does the
  retrieved text actually *contain the known evidence* for the question? (On the QASPER data,
  "evidence" is marked by human annotators; more on the proxy below.)
- The whole sweep is **deterministic, resumable, and cache-friendly**, so it runs on a free
  cloud tier.

**Why this matters:** this harness is a durable, reusable contribution on its own. It's what
made a rigorous answer *affordable*. Anyone can now test a new chunk-sizing idea against the
same yardstick cheaply.

We also framed chunking as **two independent knobs** (a small conceptual contribution):
- **Structure axis** — *where* to cut (e.g. at section headings vs. arbitrary word counts).
- **Density axis** — *how big* each piece should be. **This project is mostly about the density
  axis.**

---

## 4. What we found — the results, step by step

The investigation is organized as two hypotheses, H1 and H2.

### 4.1 H1 — "Does chunk size even matter, and does the best size vary between documents?" → **YES**

We first checked whether there's *any* opportunity here. Using the harness on **QASPER** (a set
of 50 scientific papers with 171 human-answered questions), we compared:

- **Best single fixed size** for all documents: **200 words**, scoring **0.736** on the proxy.
- **Oracle** — a magic "cheat" that somehow knows the perfect size for each document
  individually: **0.808**.

The gap is **+9.8% relative improvement** left on the table. And the best size genuinely differs
across documents — the per-document ideal was spread across *every* size from 50 to 400 words.

**Conclusion:** there *is* real room for a smart per-document method to win. H1 is the
**premise** (the opportunity exists), not the achievement.

### 4.2 H2 — "Can a cheap formula capture that opportunity?" → **NO** (the honest negative result)

This is the scientifically important part. We computed six cheap text statistics (things like
average sentence length, vocabulary richness, density of numbers/symbols) and asked: does any
of them predict the ideal chunk size?

- The **strongest** statistic (vocabulary richness) had only a **weak** correlation (−0.38 on a
  −1 to +1 scale). All others were near zero.
- When we actually *used* the best statistic to pick sizes, it **beat the fixed-size baseline in
  only 7 out of 50 random test splits (14%)** and scored *worse* on average (0.704 vs. 0.729).
- We even tried a **trained** machine-learning model on the same statistics (bending our own
  "no training" rule just to check the ceiling). **It also lost** (0.707).

**Conclusion:** cheap density features cannot capture the opportunity H1 revealed. The popular
"smaller chunks for denser text" rule is **not supported** here. A single tuned size (~200
words) is a strong, hard-to-beat baseline.

### 4.3 "Is your tool just broken?" — a positive control says no

A skeptic could object: maybe the method *would* work but our estimator is buggy, or QASPER just
has too few questions per document to measure anything. We ruled both out with a **synthetic
positive control**: we *planted* a known relationship in fake data and confirmed our exact
recipe **recovers it** (win-rate climbs to 95% as evidence grows). So the tool works — the
signal simply isn't there in real documents. To reliably win you'd need a much stronger feature
than any cheap one we have.

### 4.4 "Maybe you used the wrong features?" — smarter features also fail

Perhaps surface statistics are too shallow, and the *real* driver is a document's natural
"semantic segment size" (how long it stays on one topic). We tested a richer family of
**meaning-based** features (topic-shift rate, semantic segment length, paragraph structure,
compressibility) — still cheap, still no AI. **They failed too**, even worse than the surface
features.

The decisive nail: we computed a **label-derived diagnostic** — literally measuring the length
of the human-marked evidence passages (basically cheating, using the answer key) — and even
*that* was uncorrelated with the ideal chunk size. In other words, **the ideal size does not
depend on any property of the document itself** that we could measure. The opportunity is real
but not explainable from the document alone — it likely depends on the *questions being asked*,
which is a much bigger, different project.

---

## 5. The newest step: does this hold on a *second* kind of document? (this cycle's work)

Everything above was on **QASPER (scientific papers)**. A fair criticism: *"Maybe scientific
papers are special. Your negative result might be QASPER-specific."*

So the latest work (`notebooks/second_corpus_calibration.ipynb`) repeats the whole analysis on
a **completely different genre: QuALITY** — long **narrative stories** with multiple-choice
questions. This is deliberately as unlike scientific papers as we could reasonably get.

### 5.1 One necessary adaptation: the proxy score

QASPER comes with human-marked evidence passages; QuALITY does not. So on QuALITY we switched
to an **answer-recall proxy**: for each question, we check how much of the correct answer's text
shows up in the retrieved context. This is a **weaker, noisier** proxy (answers are often
paraphrased, not quoted), so — being honest — we first verified it isn't degenerate:

- Mean answer-recall = **0.620** (healthy — not stuck at 0 or 1).
- Median story length = **4,685 words** (long enough that chunk size is a genuine choice).

Only after confirming the proxy is meaningful do we read the result.

### 5.2 The result: the negative finding replicates

- **Opportunity (H1) is smaller here:** best fixed size 400 words (0.623) vs. oracle 0.641 —
  only **+3.0% relative headroom** (versus +9.8% on QASPER). There's simply less to win.
- **The cheap features still don't win:** the best feature reached a correlation of only 0.40
  (still too weak), and when used to pick sizes it beat the fixed baseline in **26 of 50 splits
  (52%) — a coin flip**, with essentially identical average score (0.626 vs. 0.627).

So on QuALITY the cheap-feature approach is **"non-inferior but not better"**: it ties the
tuned fixed size but never beats it. On QASPER it clearly *lost*. Either way, **the smart
per-document approach fails to beat one well-chosen global size.**

### 5.3 A genuine nuance worth noting

The **meaning-based** features carried **more** signal on narrative prose than on scientific
papers (e.g. semantic-segment length correlated −0.40 on QuALITY vs. −0.07 on QASPER). This makes
intuitive sense — "topics" are a more natural unit in stories than in dense technical papers.
But even so, the correlation topped out around 0.40, short of what's needed to win, and the small
headroom means even a perfect oracle barely beats fixed. **The overall conclusion is unchanged
across both corpora.**

### 5.4 A related control (structure axis)

Separately, on QuALITY we confirmed the *structure* axis (cutting at headings) does **no harm**
on unstructured prose: since stories have no exploitable headings, the system correctly fell
back to plain chunking on 100% of documents and matched the baseline. That's the intended
"safe fallback" behavior.

---

## 6. What this all means (the takeaways)

**For practitioners:**
> Don't bother with cheap per-document chunk-size formulas. **Tune one good global leaf size**
> (~200 words for scientific papers, ~400 for long narratives) — it is a strong, hard-to-beat
> baseline. The "smaller chunks for denser text" folklore is not supported at this level.

**For researchers:**
> This is a clean, reproducible **refutation** of a common-but-unvalidated heuristic, backed by
> (a) a learned-model ceiling that also fails, (b) a positive control proving the method isn't
> broken, (c) a label-derived diagnostic showing the ideal size isn't a function of the document
> at all, and (d) replication on a second, very different corpus. The remaining opportunity most
> likely lives in **query-aware** methods (using the questions, not just the document) — which is
> future work and a much larger undertaking.

**The lasting artifacts:**
> A **cheap, no-AI, reusable harness** for studying chunk granularity, plus a paper draft
> (`paper/main.tex`) that documents the harness, the opportunity (H1), and the refutation (H2 +
> generality).

---

## 7. Where things stand & honest limits

**Status:** the paper draft (`paper/main.tex`, ~566 lines, 14 sections) now integrates
everything above, including the second-corpus generality section. The negative result is
supported from multiple independent angles and on two corpora.

**Honest limitations (stated up front, also in the paper):**
- The scores are a **retrieval proxy**, not end-task answer quality — comparisons are
  arm-vs-arm within our frozen pipeline, not public leaderboard numbers.
- The main results are **modest scale** (50 QASPER papers, 30 QuALITY stories); confidence
  intervals are wide.
- The second-corpus result uses the **weaker answer-recall proxy** and lands at near-parity, so
  it is a **non-inferiority** generalization — it addresses the "QASPER is special" worry but is
  not a second *decisive* refutation.
- We have **not** tested **query-aware / learned multi-signal** methods, where a genuine win may
  still exist. We only refute the cheap, training-free, document-only route — which was precisely
  the contribution we set out to test.

**Natural next steps:** run the end-to-end structured-corpus check for the structure axis, and
(as a larger future project) explore query-aware granularity, where the unclaimed headroom
probably lives.

---

## 8. Pointers (the receipts)

- **Paper draft:** `paper/main.tex`
- **This cycle's notebook:** `notebooks/second_corpus_calibration.ipynb`
- **Second-corpus result (technical):** `docs/results/2026-06-30-second-corpus-generality.md`
- **QASPER H1/H2 final:** `docs/results/2026-06-15-h2-density-50doc.md`
- **"Is the tool broken?" power analysis:** `docs/results/2026-06-18-signal-recovery-power.md`
- **Smarter (locality) features:** `docs/results/2026-06-27-locality-calibration.md`
- **QuALITY structure control:** `docs/results/2026-06-11-quality-ahc-vs-token-pilot.md`
- **Prior full briefing:** `docs/2026-06-18-progress-briefing.md`
- **Harness code:** `experiments/granularity_sweep.py`, `experiments/granularity_calibration.py`
