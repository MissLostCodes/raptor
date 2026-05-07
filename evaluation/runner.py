import os
import sys
import json

# --- make sure RAPTOR is importable (works in Colab when cwd=raptor/evaluation)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from RetrievalAugmentation import RetrievalAugmentation

from datasets.qasper_loader import load_qasper
from datasets.narrativeqa_loader import load_narrativeqa
from datasets.quality_loader import load_quality

from metrics.f1 import compute_f1
from metrics.bleu_rouge_meteor import compute_metrics
from metrics.accuracy import compute_accuracy

from config import DATASET_LIMIT


# ---------- DUMMY QA MODEL ----------
def qa_model(context, question):
    return "dummy answer"  # replace with LLM later


def build_raptor(docs):
    raptor = RetrievalAugmentation()
    raptor.add_documents(docs)
    return raptor


def retrieve_context(raptor, question):
    return raptor.retrieve(question)


def evaluate_qasper(raptor, data):
    scores = []
    for item in data:
        context = retrieve_context(raptor, item["question"])
        pred = qa_model(context, item["question"])
        f1 = compute_f1(pred, item["answers"])
        scores.append(f1)
    return sum(scores) / len(scores)


def evaluate_narrativeqa(raptor, data):
    bleu_scores, rouge_scores, meteor_scores = [], [], []
    for item in data:
        context = retrieve_context(raptor, item["question"])
        pred = qa_model(context, item["question"])
        bleu, rouge, meteor = compute_metrics(pred, item["answers"])

        bleu_scores.append(bleu)
        rouge_scores.append(rouge)
        meteor_scores.append(meteor)

    return {
        "BLEU": sum(bleu_scores)/len(bleu_scores),
        "ROUGE": sum(rouge_scores)/len(rouge_scores),
        "METEOR": sum(meteor_scores)/len(meteor_scores)
    }


def evaluate_quality(raptor, data):
    scores = []
    for item in data:
        context = retrieve_context(raptor, item["question"])
        pred = 0  # dummy prediction
        acc = compute_accuracy(pred, item["answer"])
        scores.append(acc)
    return sum(scores) / len(scores)


def main():
    print("Loading datasets...")
    qasper = load_qasper(limit=DATASET_LIMIT)
    narrative = load_narrativeqa(limit=DATASET_LIMIT)
    quality = load_quality(limit=DATASET_LIMIT)

    docs = [d["context"] for d in qasper]

    print("Building RAPTOR...")
    raptor = build_raptor(docs)

    print("Running Evaluation...")

    qasper_score = evaluate_qasper(raptor, qasper)
    narrative_score = evaluate_narrativeqa(raptor, narrative)
    quality_score = evaluate_quality(raptor, quality)

    results = {
        "QASPER_F1": qasper_score,
        "NarrativeQA": narrative_score,
        "QuALITY_Accuracy": quality_score
    }

    print(results)

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Saved to results.json")


if __name__ == "__main__":
    main()