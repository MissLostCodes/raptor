from datasets import load_dataset

def load_narrativeqa(split="validation", limit=None):
    dataset = load_dataset("narrativeqa", split=split)

    data = []
    for i, item in enumerate(dataset):
        if limit and i >= limit:
            break

        context = item["document"]["summary"]["text"]

        for qa in item["questions"]:
            question = qa["question"]
            answers = [a["text"] for a in qa["answers"]]

            data.append({
                "context": context,
                "question": question,
                "answers": answers
            })

    return data
