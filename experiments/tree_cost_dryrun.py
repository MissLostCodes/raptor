"""What does it actually cost to run the granularity sweep through a REAL tree?

``granularity_sweep`` scores leaf sizes over a *flat* FAISS index -- no RAPTOR
tree is ever built (see its ``build_flat_retriever``). Rebuilding it
hierarchically costs LLM summarization calls during tree construction, and the
paper's harness section currently calls that "prohibitive on a free-tier
budget" without ever having measured it. This measures it.

The key accounting point: the coverage proxy
(``granularity_sweep.evidence_coverage``) is pure token overlap and needs NO
LLM, so going hierarchical costs *tree construction only* -- zero QA calls.
Summarization runs once per **cluster** per layer, and ``construct_tree`` stops
once a layer is down to ``reduction_dimension + 1`` (11) nodes, so the count per
document is far smaller than "one call per leaf" intuition suggests.

Offline by default: substitutes an extractive summarizer so the call count and
tree shape are real without an API key or a cent spent. ``--live`` bills the
real model (validates the OpenRouter path end-to-end; pennies for a few docs).

Run::

    python -m experiments.tree_cost_dryrun --docs 2
    python -m experiments.tree_cost_dryrun --docs 2 --live
    python -m experiments.tree_cost_dryrun --selfcheck
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from typing import List, Optional

from raptor.SummarizationModels import BaseSummarizationModel

# OpenRouter list price for paid openai/gpt-oss-120b, checked 2026-07-15.
# ponytail: hardcoded; re-check before quoting these numbers in the paper.
PRICE_IN_PER_M = 0.03
PRICE_OUT_PER_M = 0.15

# OpenRouter free-tier ceiling: 20 req/min, 1000 req/day once $10 of credit has
# been purchased (50/day before that).
FREE_REQS_PER_DAY = 1000

# The sweep grid from granularity_sweep / the paper.
SIZES = [50, 100, 150, 200, 300, 400]


def _tokenizer():
    """cl100k_base -- the same encoding ``tree_builder`` counts with."""
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


class CountingSummarizer(BaseSummarizationModel):
    """Tally calls and tokens through a summarization model.

    ``TreeBuilder.summarize`` calls ``summarization_model.summarize(context,
    max_tokens)`` positionally, so that is the signature we mirror. Subclasses
    ``BaseSummarizationModel`` because ``TreeBuilder`` isinstance-checks it.
    """

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0
        self.in_tokens = 0
        self.out_tokens = 0
        self._tok = _tokenizer()

    def summarize(self, context, max_tokens=150, stop_sequence=None) -> str:
        out = self.inner.summarize(context, max_tokens)
        self.calls += 1
        self.in_tokens += len(self._tok.encode(context or ""))
        self.out_tokens += len(self._tok.encode(out or ""))
        return out


class OfflineSummarizer(BaseSummarizationModel):
    """Free stand-in for the real model: a leading extractive slice.

    Reuses ``models._extractive_summary_fallback``. Summaries must stay
    *textually distinct* across clusters: a constant stub would give every
    parent node the same embedding, collapse the next layer into one cluster,
    and undercount the calls we are here to measure. An extractive slice keeps
    the tree's shape realistic.
    """

    def summarize(self, context, max_tokens=150, stop_sequence=None) -> str:
        from experiments.models import _extractive_summary_fallback

        return _extractive_summary_fallback(context, max_tokens)


def _null_qa_model():
    """A BaseQAModel that must never be called.

    ``RetrievalAugmentationConfig`` defaults a missing ``qa_model`` to
    ``GPT3TurboQAModel()``, which would demand an OpenAI key we neither have nor
    need -- the dry run only builds trees, it never answers.
    """
    from raptor.QAModels import BaseQAModel

    class _NullQAModel(BaseQAModel):
        def answer_question(self, context, question, *args, **kwargs):
            raise AssertionError("QA model called during a tree cost dry run")

    return _NullQAModel()


def probe_document(doc, sizes, summarizer, cfg, qa_model=None) -> List[dict]:
    """Build ``doc``'s tree at each leaf size; return per-size cost + shape (HEAVY).

    One row per size: the summarization calls and tokens that size's tree cost,
    plus ``n_leaves`` / ``n_layers``. The shape columns are the point as much as
    the cost ones -- leaf size drives tree depth (``construct_tree`` stops at
    <= 11 nodes), so at coarse sizes a short document may build barely any tree
    at all, which is exactly the interaction the flat proxy cannot see.

    Shares ONE SBERT across every build via ``granularity_sweep.get_shared_embedder``
    -- the same singleton the flat sweep uses. Without it each of the ~30 builds
    constructs its own 438 MB model and the T4 OOMs partway through.
    """
    from experiments.granularity_sweep import get_shared_embedder
    from experiments.pipeline import build_answerer

    qa_model = qa_model or _null_qa_model()
    embedder = get_shared_embedder()
    rows: List[dict] = []
    for size in sizes:
        before = (summarizer.calls, summarizer.in_tokens, summarizer.out_tokens)
        cfg_size = dataclasses.replace(cfg, leaf_max_tokens=size)
        answerer = build_answerer(
            "token", doc, cfg_size, summarizer, qa_model, embedding_model=embedder
        )
        tree = getattr(getattr(answerer, "_ra", None), "tree", None)
        rows.append(
            {
                "doc_id": doc.doc_id,
                "size": size,
                "calls": summarizer.calls - before[0],
                "in_tokens": summarizer.in_tokens - before[1],
                "out_tokens": summarizer.out_tokens - before[2],
                "n_leaves": len(getattr(tree, "leaf_nodes", None) or {}),
                "n_layers": getattr(tree, "num_layers", None),
            }
        )
    return rows


def project(rows: List[dict], n_probed: int, n_target: int) -> dict:
    """Extrapolate measured ``rows`` from ``n_probed`` documents to ``n_target``.

    Linear in document count: every document pays its own tree construction, and
    the DiskCache only dedupes *identical* summarization contexts, which across
    distinct papers is ~never. Assumes the probed documents are length-typical --
    probe enough of them that they are.
    """
    if n_probed <= 0:
        raise ValueError("n_probed must be positive")
    scale = n_target / n_probed
    calls = sum(r["calls"] for r in rows) * scale
    in_tokens = sum(r["in_tokens"] for r in rows) * scale
    out_tokens = sum(r["out_tokens"] for r in rows) * scale
    usd = in_tokens / 1e6 * PRICE_IN_PER_M + out_tokens / 1e6 * PRICE_OUT_PER_M
    return {
        "n_docs": n_target,
        "calls": round(calls),
        "in_tokens": round(in_tokens),
        "out_tokens": round(out_tokens),
        "usd_paid": round(usd, 2),
        "free_tier_days": round(calls / FREE_REQS_PER_DAY, 1),
    }


def _print_report(rows: List[dict], n_probed: int, targets: List[int], live: bool) -> None:
    by_size: dict = {}
    for r in rows:
        agg = by_size.setdefault(r["size"], {"calls": 0, "leaves": 0, "layers": []})
        agg["calls"] += r["calls"]
        agg["leaves"] += r["n_leaves"]
        if r["n_layers"] is not None:
            agg["layers"].append(r["n_layers"])

    mode = "LIVE (billed)" if live else "OFFLINE (extractive stand-in)"
    print(f"\n=== tree cost dry run: {n_probed} doc(s), {mode} ===\n")
    print(f"{'leaf size':>10} {'calls/doc':>10} {'leaves/doc':>11} {'layers':>8}")
    for size in sorted(by_size):
        agg = by_size[size]
        layers = agg["layers"]
        mean_layers = f"{sum(layers) / len(layers):.1f}" if layers else "-"
        print(
            f"{size:>10} {agg['calls'] / n_probed:>10.1f} "
            f"{agg['leaves'] / n_probed:>11.1f} {mean_layers:>8}"
        )

    total = sum(r["calls"] for r in rows)
    print(f"\ntotal calls for all {len(SIZES)} sizes: {total / n_probed:.1f} per document\n")
    print(f"{'target':>8} {'calls':>9} {'paid USD':>10} {'free-tier days':>15}")
    for t in targets:
        p = project(rows, n_probed, t)
        print(f"{p['n_docs']:>8} {p['calls']:>9} {p['usd_paid']:>10.2f} {p['free_tier_days']:>15}")
    print()


def selfcheck() -> None:
    """Offline asserts for the two pieces of non-trivial logic (no network)."""

    class _Echo:
        def summarize(self, context, max_tokens=150, stop_sequence=None):
            return "word " * 10

    c = CountingSummarizer(_Echo())
    c.summarize("alpha beta gamma", 150)
    c.summarize("delta", 150)
    assert c.calls == 2, c.calls
    assert c.in_tokens > 0 and c.out_tokens > 0
    # The tally is per-call and cumulative, not per-document.
    before = c.in_tokens
    c.summarize("epsilon", 150)
    assert c.calls == 3 and c.in_tokens > before

    # Projection scales linearly and prices both token directions.
    rows = [{"calls": 10, "in_tokens": 1_000_000, "out_tokens": 1_000_000}]
    p = project(rows, n_probed=1, n_target=2)
    assert p["calls"] == 20, p
    assert p["in_tokens"] == 2_000_000 and p["out_tokens"] == 2_000_000
    # 2M in @ $0.03/M + 2M out @ $0.15/M = 0.06 + 0.30 = 0.36
    assert p["usd_paid"] == 0.36, p
    assert p["free_tier_days"] == round(20 / FREE_REQS_PER_DAY, 1), p

    # Identity scale is a no-op; a zero probe is a hard error, not a div-by-zero.
    assert project(rows, 1, 1)["calls"] == 10
    try:
        project(rows, 0, 50)
    except ValueError:
        pass
    else:
        raise AssertionError("n_probed=0 must raise")

    print("selfcheck OK")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Measure RAPTOR tree-build LLM cost.")
    ap.add_argument("--docs", type=int, default=2, help="documents to probe")
    ap.add_argument("--live", action="store_true", help="bill the real model")
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--targets", type=int, nargs="+", default=[50, 416])
    ap.add_argument("--json-out", type=str, default=None)
    ap.add_argument("--selfcheck", action="store_true", help="run asserts and exit")
    args = ap.parse_args(argv)

    if args.selfcheck:
        selfcheck()
        return

    from experiments.config import ExperimentConfig
    from experiments.datasets.qasper import QASPERLoader

    cfg = ExperimentConfig()
    docs = QASPERLoader(split="test").load(limit=args.docs)
    if not docs:
        raise SystemExit("no documents loaded")

    if args.live:
        from experiments.models import build_models

        inner, qa_model = build_models(cfg)
    else:
        inner, qa_model = OfflineSummarizer(), None

    summarizer = CountingSummarizer(inner)
    rows: List[dict] = []
    for i, doc in enumerate(docs, 1):
        print(f"[dryrun] doc {i}/{len(docs)} {doc.doc_id}: building {len(args.sizes)} trees")
        rows.extend(probe_document(doc, args.sizes, summarizer, cfg, qa_model))

    _print_report(rows, len(docs), args.targets, args.live)

    if args.json_out:
        payload = {
            "rows": rows,
            "n_probed": len(docs),
            "live": args.live,
            "projections": [project(rows, len(docs), t) for t in args.targets],
        }
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
