def compute_accuracy(pred, gold):
    return 1 if str(pred).strip() == str(gold).strip() else 0
