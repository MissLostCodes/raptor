def compute_f1(pred, gold_list):
    def f1_single(pred, gold):
        pred_tokens = pred.lower().split()
        gold_tokens = gold.lower().split()

        common = set(pred_tokens) & set(gold_tokens)
        if len(common) == 0:
            return 0

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)

        return 2 * precision * recall / (precision + recall)

    return max(f1_single(pred, g) for g in gold_list)
