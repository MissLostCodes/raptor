"""One-shot builder for notebooks/granularity_calibration.ipynb (deleted after use)."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []

# 1. md -- purpose
cells.append(new_markdown_cell(
    "# Granularity Calibration — adaptive headroom + training-free density→size map\n"
    "\n"
    "**H1 confirmed** (the prior sweep showed the optimal QASPER leaf size is "
    "document-dependent, spanning 50–400 tokens). Here we:\n"
    "\n"
    "- **(a)** quantify the **adaptive headroom** an oracle sizing policy has over the "
    "single best fixed leaf size,\n"
    "- **(b)** calibrate a **training-free density→size map** and evaluate it on a "
    "**held-out** split of documents,\n"
    "- **(c)** contrast that parsimonious 2-parameter map with a **learned** "
    "multi-feature regressor.\n"
    "\n"
    "**SBERT is used only for the sweep** (to measure evidence coverage); there is "
    "**no LLM** anywhere in this notebook. The density score is deterministic and "
    "training-free, and the analytical functions in `experiments.granularity_calibration` "
    "are pure stdlib (unit-tested offline)."
))

# 2. code -- clone / install / Drive / OUT
cells.append(new_code_cell(
    "!git clone -b ckraptor https://github.com/MissLostCodes/raptor.git\n"
    "%cd raptor\n"
    "!pip install -q -r requirements-colab.txt"
))
cells.append(new_code_cell(
    "# Persist the sweep output to Drive so a disconnect never loses progress; rerun resumes.\n"
    "try:\n"
    "    from google.colab import drive\n"
    "    drive.mount('/content/drive')\n"
    "    RUN_DIR = '/content/drive/MyDrive/raptor_runs'\n"
    "except Exception as e:\n"
    "    print('Not on Colab / Drive unavailable -> using local ./raptor_runs', e)\n"
    "    RUN_DIR = 'raptor_runs'\n"
    "\n"
    "import os\n"
    "os.makedirs(RUN_DIR, exist_ok=True)\n"
    "# Reuse the SAME sweep cache the GO/NO-GO notebook writes; '_50' marks the 50-doc run.\n"
    "OUT = os.path.join(RUN_DIR, 'granularity_sweep_50.json')\n"
    "print('RUN_DIR =', RUN_DIR)\n"
    "print('OUT     =', OUT)"
))

# 3. code -- load docs
cells.append(new_code_cell(
    "from experiments.datasets import get_loader\n"
    "\n"
    "docs = get_loader('qasper').load(limit=50)\n"
    "n_ans = sum(1 for d in docs for q in d.questions if q.evidence)\n"
    "print(f'{len(docs)} QASPER papers, {n_ans} answerable (gold-evidence) questions total')"
))

# 4. code -- run sweep (resumable)
cells.append(new_code_cell(
    "from experiments import granularity_sweep as gs, granularity_calibration as gc\n"
    "\n"
    "SIZES = [50, 100, 150, 200, 300, 400]   # leaf token sizes to sweep\n"
    "# SBERT-only, NO LLM. Resumable from OUT: first run can take a while (embeds every\n"
    "# chunk of every paper at every size); reruns are fast (cached records are skipped).\n"
    "records = gs.run_sweep(docs, SIZES, budget=2000, out_path=OUT)\n"
    "print(f'{len(records)} (doc, size) records')\n"
    "records[:3]"
))

# 5. code -- HEADROOM + Fig 1
cells.append(new_code_cell(
    "import matplotlib.pyplot as plt\n"
    "\n"
    "# Fig.1 + the headroom number: how much an adaptive policy could gain over the\n"
    "# single best fixed leaf size.\n"
    "h = gc.headroom(records)\n"
    "print('HEADROOM:', h)\n"
    "print(\n"
    "    f\"best fixed size = {h['best_fixed_size']} tok @ cov {h['best_fixed_cov']:.4f} | \"\n"
    "    f\"oracle cov {h['oracle_cov']:.4f} | abs_gap {h['abs_gap']:.4f} \"\n"
    "    f\"(+{h['rel_gain_pct']:.1f}% rel.)\"\n"
    ")\n"
    "\n"
    "# Mean coverage vs fixed leaf size, with the per-doc oracle as a horizontal ceiling.\n"
    "sizes_sorted = sorted({r['size'] for r in records if r['mean_evidence_coverage'] is not None})\n"
    "fixed_curve = [gc.mean_coverage_at_size(records, s) for s in sizes_sorted]\n"
    "\n"
    "plt.figure(figsize=(7, 4))\n"
    "plt.plot(sizes_sorted, fixed_curve, marker='o', label='mean coverage @ fixed size')\n"
    "plt.axhline(h['oracle_cov'], color='C3', ls='--',\n"
    "            label=f\"per-doc oracle = {h['oracle_cov']:.3f}\")\n"
    "plt.scatter([h['best_fixed_size']], [h['best_fixed_cov']], color='C1', zorder=5,\n"
    "            label=f\"best fixed = {h['best_fixed_size']} tok\")\n"
    "plt.xlabel('leaf chunk size (tokens)')\n"
    "plt.ylabel('mean evidence coverage')\n"
    "plt.title('Fig.1 — adaptive headroom: best fixed size vs per-doc oracle')\n"
    "plt.grid(True, alpha=0.3)\n"
    "plt.legend(fontsize=8)\n"
    "plt.show()"
))

# 6. code -- density features per doc + optimal sizes
cells.append(new_code_cell(
    "from raptor.chunking.density_score import density_score\n"
    "\n"
    "# Deterministic, training-free density signal per document (rho + its 6 features).\n"
    "feats = {d.doc_id: density_score(d.text)[1] for d in docs}\n"
    "rho = {d.doc_id: density_score(d.text)[0] for d in docs}\n"
    "\n"
    "# Per-doc optimal leaf size from the sweep (the calibration target).\n"
    "opt = gs.best_size_per_doc(records)\n"
    "print(f'{len(opt)} docs with a defined optimal size')\n"
    "from collections import Counter\n"
    "print('distribution of optimal sizes:', dict(Counter(opt.values())))"
))

# 7. code -- CORRELATION of rho + each feature vs optimal size
cells.append(new_code_cell(
    "# Which density signal predicts granularity? Spearman rank-corr of rho and each of\n"
    "# the 6 density features against the per-doc optimal leaf size (feature ablation).\n"
    "doc_ids = [k for k in opt if k in rho]   # docs with both a target and a score\n"
    "\n"
    "FEATURE_NAMES = [\n"
    "    'mean_sentence_len_tokens_norm',\n"
    "    'type_token_ratio',\n"
    "    'numeral_symbol_density',\n"
    "    'mean_word_len_chars_norm',\n"
    "    'list_marker_density',\n"
    "    'nonstopword_ratio',\n"
    "]\n"
    "\n"
    "opt_vec = [opt[k] for k in doc_ids]\n"
    "print(f\"{'signal':<32}{'spearman vs opt size':>22}\")\n"
    "print('-' * 54)\n"
    "rc_rho = gc.rank_corr([rho[k] for k in doc_ids], opt_vec)\n"
    "print(f\"{'rho (combined density)':<32}{rc_rho:>22.3f}\")\n"
    "for f in FEATURE_NAMES:\n"
    "    rc = gc.rank_corr([feats[k][f] for k in doc_ids], opt_vec)\n"
    "    print(f'{f:<32}{rc:>22.3f}')"
))

# 8. code -- CALIBRATE on train, EVALUATE on held-out test (Table 1)
cells.append(new_code_cell(
    "# Paper Table 1: held-out evaluation of the training-free density->size map.\n"
    "tr, te = gc.train_test_split_docs(list(opt))\n"
    "print(f'{len(tr)} train docs, {len(te)} test docs')\n"
    "\n"
    "# Fit the parsimonious size = a + b*rho on TRAIN only.\n"
    "fit = gc.fit_rho_to_size({k: rho[k] for k in tr}, {k: opt[k] for k in tr})\n"
    "print('fit:', fit)\n"
    "\n"
    "# Density-predicted leaf sizes for the TEST docs (clipped to the swept range).\n"
    "pred_test = {k: gc.predict_size(rho[k], fit, l_min=50, l_max=400) for k in te}\n"
    "\n"
    "# Score each policy on the TEST docs (coverage snapped to the swept grid).\n"
    "best_fixed_size, _ = gc.best_global_fixed(records)\n"
    "cov_fixed   = gc.coverage_under_sizes(records, {k: best_fixed_size for k in te})\n"
    "cov_density = gc.coverage_under_sizes(records, pred_test)\n"
    "_, oracle_sizes = gc.oracle_per_doc(records)\n"
    "cov_oracle  = gc.coverage_under_sizes(records, {k: oracle_sizes[k] for k in te if k in oracle_sizes})\n"
    "\n"
    "print()\n"
    "print(f\"{'policy (TEST docs)':<34}{'mean coverage':>14}\")\n"
    "print('-' * 48)\n"
    "print(f\"{'best global fixed (' + str(best_fixed_size) + ' tok)':<34}{cov_fixed:>14.4f}\")\n"
    "print(f\"{'density-predicted size (rho->size)':<34}{cov_density:>14.4f}\")\n"
    "print(f\"{'per-doc oracle (ceiling)':<34}{cov_oracle:>14.4f}\")"
))

# 9. code -- LEARNED baseline (sklearn)
cells.append(new_code_cell(
    "# Learned baseline: a 6-feature linear regressor (sklearn) fit on TRAIN, evaluated\n"
    "# on TEST the same way. Shows whether the training-free 2-param map is competitive\n"
    "# with a fully learned multi-feature model.\n"
    "from sklearn.linear_model import LinearRegression\n"
    "\n"
    "X_tr = [[feats[k][f] for f in FEATURE_NAMES] for k in tr]\n"
    "y_tr = [opt[k] for k in tr]\n"
    "reg = LinearRegression().fit(X_tr, y_tr)\n"
    "\n"
    "def _clip_size(v, lo=50, hi=400):\n"
    "    return int(max(lo, min(hi, round(v))))\n"
    "\n"
    "pred_test_learned = {\n"
    "    k: _clip_size(reg.predict([[feats[k][f] for f in FEATURE_NAMES]])[0]) for k in te\n"
    "}\n"
    "cov_learned = gc.coverage_under_sizes(records, pred_test_learned)\n"
    "\n"
    "print(f\"{'policy (TEST docs)':<34}{'mean coverage':>14}\")\n"
    "print('-' * 48)\n"
    "print(f\"{'best global fixed':<34}{cov_fixed:>14.4f}\")\n"
    "print(f\"{'density-predicted (training-free)':<34}{cov_density:>14.4f}\")\n"
    "print(f\"{'learned regressor (sklearn, 6 feat)':<34}{cov_learned:>14.4f}\")\n"
    "print(f\"{'per-doc oracle (ceiling)':<34}{cov_oracle:>14.4f}\")"
))

# 10. md -- reading guide
cells.append(new_markdown_cell(
    "## How to read this\n"
    "\n"
    "- **Headroom (Fig.1).** If the per-doc oracle sits well above the best fixed-size "
    "point, there is real coverage to be won by sizing leaves per document. A negligible "
    "`abs_gap` means a fixed size is already near-optimal and the adaptive story is weak.\n"
    "- **Correlation.** The signal(s) with the largest-magnitude Spearman vs optimal size "
    "are the ones that actually carry granularity information; near-zero rows are dead "
    "features for sizing. The sign of `rho`'s correlation tells you the map's direction "
    "(negative ⇒ denser text wants smaller leaves — the `invert=True` hypothesis).\n"
    "- **Adaptive wins** when, on the **held-out TEST** docs, the **density-predicted** "
    "coverage **exceeds best global fixed** and **approaches the per-doc oracle**.\n"
    "- **Training-free ≈ learned** when the parsimonious `size = a + b·rho` map lands "
    "close to the 6-feature sklearn regressor on TEST. If so, the cheap, interpretable, "
    "label-free map is the one to ship — no per-corpus training required at inference.\n"
    "\n"
    "Scale up `limit=` if the picture is borderline; the sweep is cached in `OUT` and "
    "resumes, so re-running only embeds the newly added documents."
))

nb = new_notebook(cells=cells)
nb.metadata["colab"] = {"provenance": []}
nb.metadata["kernelspec"] = {
    "display_name": "Python 3", "language": "python", "name": "python3",
}
nb.metadata["language_info"] = {"name": "python"}
nb.nbformat = 4
nb.nbformat_minor = 5

out = "notebooks/granularity_calibration.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(nb.cells), "cells")
