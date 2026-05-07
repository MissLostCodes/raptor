from datasets import load_dataset

def load_quality(split="validation", limit=None):
    dataset = load_dataset("quality", split=split)

    data = []
    for i, item in enumerate(dataset):
        if limit and i >= limit:
            break

        context = item["article"]

        for q in item["questions"]:
            question = q["question"]
            options = q["options"]
            label = q["gold_label"]

            data.append({
                "context": context,
                "question": question,
                "options": options,
                "answer": label
            })

    return data
