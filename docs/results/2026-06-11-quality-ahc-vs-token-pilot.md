# Quality pilot: AHC vs token leaf-chunking on QuALITY

**Date:** 2026-06-11
**Dataset:** QuALITY (multiple-choice reading comprehension over narrative prose; accuracy, plus a QuALITY-HARD subset; chance = 25%)
**Arms:** `token` (baseline RAPTOR sentence/token splitter) vs `ahc` (Adaptive Hybrid Chunking; a deterministic structure score routes each doc to structure-aware vs token chunking, threshold tau = 0.5)
**Pipeline:** Otherwise frozen — same `gpt-oss-120b` reader for summarization and QA across both arms, so only the leaf chunker varies.
**Setup:** Free-tier Cerebras pilot, 6 documents.

## TL;DR

This is a **statistical tie**, and the reason is mechanical rather than a property of the chunkers: **AHC never engaged its structure path on QuALITY.** The structure-routing rate was `0.0000` over all 6 documents, so AHC fell back to plain token chunking on every doc — i.e. it ran *the same* chunker as the `token` arm. The ~2-question accuracy difference that remains is residual pipeline nondeterminism (RAPTOR clustering + LLM summarization + generation), not a chunking effect. QuALITY is plain narrative prose with no headings or sections, so it functions here as an **unstructured control**: the right takeaway is that AHC correctly does no harm when there is no structure to exploit, not that AHC was tested.

## Results

| arm   | metric         |   mean |   std |   n | 95% CI             |
|-------|----------------|-------:|------:|----:|--------------------|
| ahc   | accuracy       | 0.4000 | 0.4922 | 105 | [0.3143, 0.5048]   |
| ahc   | accuracy_hard  | 0.3729 | 0.4877 |  59 | [0.2542, 0.4915]   |
| token | accuracy       | 0.3810 | 0.4880 | 105 | [0.2952, 0.4857]   |
| token | accuracy_hard  | 0.3559 | 0.4829 |  59 | [0.2373, 0.4915]   |

**In raw questions (concrete view):**

- **Overall:** AHC `0.4000 × 105 = 42` correct vs token `0.3810 × 105 = 40` correct → a **2-question gap** over 105 paired questions.
- **Hard subset:** AHC `0.3729 × 59 = 22` correct vs token `0.3559 × 59 = 21` correct → a **1-question gap** over 59 paired questions.

The 95% confidence intervals overlap almost entirely on both metrics.

**Diagnostics:**

- AHC structure-routing rate: `0.0000` over `n_docs = 6` (no document routed to the structure chunker).
- Dropped (moderation-blocked) questions: `4`.

## Root cause: AHC never ran its structure path

AHC routes a document to its structure-aware chunker only when a deterministic structure score is `>= tau (0.5)`; otherwise it falls back to token chunking. QuALITY documents are plain narrative prose with no headings or sections, so the structure score was **0 for all 6 documents**. As a result:

- **`structure_rate = 0.0000`** — every document fell back to token chunking.
- The `ahc` arm therefore executed **the exact same** token chunking as the `token` arm.
- The 2-question overall difference (and 1-question hard-subset difference) is **residual pipeline nondeterminism** — RAPTOR's clustering step plus LLM summarization and answer generation are not bit-for-bit deterministic across runs — **not a chunking effect.**

In other words, this pilot did not actually exercise AHC's structure path. It is a same-chunker-vs-same-chunker comparison, which is exactly why the two arms land on top of each other.

## Is it significant?

No. This is a **paired comparison**: the harness uniformly drops any question that is blocked under *any* arm, so both arms are scored on the **identical 105 questions** (and identical 59 on the hard subset). The appropriate test for two binary-outcome systems on the same items is a **paired (McNemar) test**, not a comparison of independent-sample means.

A net difference of **2 questions out of 105** (and **1 out of 59** on the hard subset) is nowhere near significant under any reasonable paired test, and the overlapping 95% CIs corroborate this at a glance. Treat the arms as indistinguishable on this dataset.

## Caveats and paper comparison

- **Small N.** This is a 6-document free-tier pilot. N is small and the CIs are correspondingly wide (roughly +/-0.10 to +/-0.12 on accuracy). This is a **signal-check, not a final number.**
- **Unbiased dropping.** ~4 of ~109 QuALITY questions were moderation-blocked and removed **uniformly from both arms**, leaving `n = 105` paired. This is minor and unbiased — the comparison stays fair because both arms lose the same questions.
- **Reference band (for orientation only — NOT a target).** From the RAPTOR paper (dev split, **different readers**, so not directly comparable):
  - BM25 + UnifiedQA: 49.9
  - DPR + UnifiedQA: 53.9
  - RAPTOR + UnifiedQA-3B: 56.6
  - RAPTOR + GPT-3: 62.4 (dev)

  These are listed purely to situate the metric. **Do not** compare against the 82.6 GPT-4 test-set headline — different split, different reader, different conditions.

## Next step → QASPER

QuALITY is the **wrong dataset to test AHC** because it has no structure to exploit; it serves only as an unstructured control (AHC correctly does no harm here). To actually exercise AHC's structure path we move to **QASPER** (scientific papers with real sections).

This required a fix first. The structure detector originally only recognized **numbered**, **markdown**, **ALL-CAPS**, and **keyword** headings — and it **missed QASPER's plain Title-Case section names** (e.g. "Related Work", "Experimental Setup"). Without a fix, AHC would have silently fallen back to token chunking on QASPER too, reproducing this same null result. A **generic Title-Case heading detector** was added so that AHC genuinely engages structure on QASPER.

The QASPER run lives in `notebooks/qasper_ahc_vs_token.ipynb`.
