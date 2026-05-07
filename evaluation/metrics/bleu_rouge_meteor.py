from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def compute_metrics(pred, gold_list):
    bleu = max(sentence_bleu([g.split()], pred.split()) for g in gold_list)
    meteor = max(meteor_score([g], pred) for g in gold_list)
    rouge = max(scorer.score(g, pred)['rougeL'].fmeasure for g in gold_list)

    return bleu, rouge, meteor
