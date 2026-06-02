# Pinned evaluation subsets

These JSON files pin the **version-controlled evaluation subsets** used by the
RAPTOR reproduction harness, so that every run scores the exact same documents
and results stay comparable across machines and over time.

Each file has the shape:

```json
{"seed": 0, "ids": []}
```

- `seed` — the seed passed to `experiments.datasets.base.select_subset`.
- `ids` — the selected `doc_id`s for that dataset. **Currently empty
  placeholders**; they are populated later by running `select_subset` on the
  full dataset (e.g. on Colab where the HuggingFace download is available) and
  committing the resulting ids here.

To (re)generate, load the full dataset with the relevant loader, call
`select_subset(documents, n, seed=<seed>, strata_fn=...)`, and write the
returned ids back into the corresponding file.

Files:

- `qasper_ids.json`      — QASPER (NLP-paper QA).
- `quality_ids.json`     — QuALITY (multiple-choice over articles).
- `narrativeqa_ids.json` — NarrativeQA (full-story abstractive QA).
