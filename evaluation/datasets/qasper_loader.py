from datasets import load_dataset

def load_qasper(split="validation", limit=None):
    dataset = load_dataset("qasper", split=split)

    data = []
    for i, item in enumerate(dataset):
        if limit and i >= limit:
            break

        context = item["full_text"]["paragraphs"]
        context = " ".join(context)

        for qa in item["qas"]:
            question = qa["question"]
            answers = qa["answers"]

            gold = []
            for ans in answers:
                if ans["answer"]["extractive_spans"]:
                    gold.append(" ".join(ans["answer"]["extractive_spans"]))
                elif ans["answer"]["free_form_answer"]:
                    gold.append(ans["answer"]["free_form_answer"])

            if gold:
                data.append({
                    "context": context,
                    "question": question,
                    "answers": gold
                })

    return data
