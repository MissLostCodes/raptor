# RAPTOR Chunking-Arms Experiment Harness

Isolates **leaf chunking** as the only variable inside an otherwise-frozen RAPTOR pipeline and
compares 5 arms on 3 QA benchmarks, using the **same gpt-oss model** (via OpenRouter) for
summarization and QA across every arm. Faithful to RAPTOR (arXiv:2401.18059v1).

## Arms
`token` (baseline RAPTOR) · `structure` (always-LLM) · **`ahc`** (Adaptive Hybrid Chunking, our
method) · `semantic` (SBERT-boundary) · `flat` (token chunks + FAISS, no tree).

## Datasets → metrics
| Dataset | HF source | Metric |
|---|---|---|
| QASPER | `allenai/qasper` | Answer token-F1 (Evidence-F1 scaffolded) |
| QuALITY | `emozilla/quality` | Accuracy (+ HARD subset) |
| NarrativeQA | `deepmind/narrativeqa` | ROUGE-L, BLEU-1/4, METEOR (AllenNLP mods) |

Retrieval: collapsed tree, 2000-token budget (the paper's main setting).

## Layout
```
experiments/
  config.py      ExperimentConfig (model, arms, datasets, subset_sizes, seeds, budgets, tau, cache)
  models.py      cached + back-off OpenRouter summarization/QA models (DiskCache, build_models)
  datasets/      base.py + qasper/quality/narrativeqa loaders + pure parsers + subsets/*.json
  metrics/       qa_f1, evidence_f1, accuracy, generation (rouge/bleu/meteor)
  pipeline.py    arm -> RetrievalAugmentation / FAISS Answerer (make_ra_config seam)
  runner.py      dataset -> QA prompt + scorer; orchestration loop -> results_<seed>.json
  report.py      aggregate (mean±std, bootstrap CI), QuALITY HARD, AHC routing rate, markdown
  run.py         CLI; select_subsets.py pins version-controlled eval subsets
```

## Run on Colab (recommended)
Open `notebooks/run_experiments.ipynb`: it clones the repo, installs `requirements-colab.txt`,
takes your OpenRouter key, runs a 3-doc pilot, prints per-arm tables, then scales up. Every LLM
call is cached on disk so reruns are cheap and resumable.

## Run from the CLI
```bash
python -m experiments.run \
  --arms token,structure,ahc,semantic,flat \
  --datasets qasper,quality,narrativeqa \
  --model openai/gpt-oss-120b:free \
  --n-qasper 50 --n-quality 50 --n-narrativeqa 25 \
  --seed 0 --results-dir results
```
Swap `--model openai/gpt-oss-120b` (paid) if the free tier rate-limits the run.

## Develop / test locally (no GPU stack needed)
The library lazy-loads torch/sentence-transformers/faiss/umap, so the full **offline** suite runs
in a light CPU-only venv (LLM + embeddings injected; no model downloads):
```bash
uv venv .venv --python 3.12
uv pip install --python .venv pytest tiktoken "openai>=1.3" numpy scikit-learn \
    rouge-score nltk sacrebleu "datasets>=2.16" evaluate tenacity
.venv/bin/python -m pytest tests/ -q
```
The heavy tree-build / FAISS paths (`pipeline.build_answerer`, `runner.run`'s real path) are
exercised on Colab, not in the unit suite.

## Reproducibility
Temperature 0 + disk cache neutralize free-tier nondeterminism; fixed seeds; frozen
version-controlled subset id lists (`datasets/subsets/*.json`, pinned via
`python -m experiments.select_subsets`); pinned model `openai/gpt-oss-120b:free`.
