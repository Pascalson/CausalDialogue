import nltk
from nltk import ngrams
from nltk.tokenize import word_tokenize

from collections import defaultdict
from itertools import chain
import sys

with open('test_v1_dialog_history.txt','r') as fin:
    DHs = fin.readlines()

with open('test_v1_gt_response.txt','r') as fin:
    y_trues = fin.readlines()

with open(sys.argv[1], 'r') as fin:
    y_preds = fin.readlines()

speaker_separator = ':'


multi_y_trues = defaultdict(list)
pure_yps = []

spk_corr = 0
for DH, yp, yt in zip(DHs, y_preds, y_trues):
    if speaker_separator in yt:
        spk_t, pure_yt = yt.split(speaker_separator,1)
    else:
        spk_t = ''
        pure_yt = yt

    if speaker_separator in yp:
        spk_p, pure_yp = yp.split(speaker_separator,1)
    else:
        spk_p = ''
        pure_yp = yp
    if spk_t == spk_p:
        spk_corr += 1

    multi_y_trues[DH].append(word_tokenize(pure_yt.strip()))
    pure_yps.append(word_tokenize(pure_yp.strip()))

chencherry = nltk.translate.bleu_score.SmoothingFunction()
macro_bleu1, macro_bleu2, macro_bleu4 = [], [], []
corpus_bleu_refs = []
heldout_corpus_bleu_refs = []
heldout_yps = []
for DH, yp in zip(DHs, pure_yps):
    corpus_bleu_refs.append(multi_y_trues[DH])#held-in
    heldout_refs = [y for y in multi_y_trues[DH] if y != yp]
    if len(heldout_refs) > 2:
        heldout_corpus_bleu_refs.append([y for y in multi_y_trues[DH] if y != yp])
        heldout_yps.append(yp)
    macro_bleu1.append(nltk.translate.bleu_score.sentence_bleu(multi_y_trues[DH], yp, weights=[1,0,0,0], smoothing_function=chencherry.method1))
    macro_bleu2.append(nltk.translate.bleu_score.sentence_bleu(multi_y_trues[DH], yp, weights=[0.5,0.5,0,0], smoothing_function=chencherry.method1))
    macro_bleu4.append(nltk.translate.bleu_score.sentence_bleu(multi_y_trues[DH], yp, smoothing_function=chencherry.method1))

micro_bleu1 = nltk.translate.bleu_score.corpus_bleu(corpus_bleu_refs, pure_yps, weights=[1,0,0,0])
micro_bleu2 = nltk.translate.bleu_score.corpus_bleu(corpus_bleu_refs, pure_yps, weights=[0.5,0.5,0,0])
micro_bleu4 = nltk.translate.bleu_score.corpus_bleu(corpus_bleu_refs, pure_yps)


heldout_micro_bleu1 = nltk.translate.bleu_score.corpus_bleu(heldout_corpus_bleu_refs, heldout_yps, weights=[1,0,0,0])
heldout_micro_bleu2 = nltk.translate.bleu_score.corpus_bleu(heldout_corpus_bleu_refs, heldout_yps, weights=[0.5,0.5,0,0])
heldout_micro_bleu4 = nltk.translate.bleu_score.corpus_bleu(heldout_corpus_bleu_refs, heldout_yps)

# for distinct-N
ngrams1 = list(chain(*[[gram for gram in ngrams(y, 1)] for y in pure_yps]))
ngrams2 = list(chain(*[[gram for gram in ngrams(y, 2)] for y in pure_yps]))
distinct1 = len(set(ngrams1)) / len(ngrams1)
distinct2 = len(set(ngrams2)) / len(ngrams2)


# only used for ground-truths
print("Heldout Micro Bleu")
print(heldout_micro_bleu1)
print(heldout_micro_bleu2)
print(heldout_micro_bleu4)

print("Micro Bleu")
print(micro_bleu1)
print(micro_bleu2)
print(micro_bleu4)

print("Macro Bleu")
print(sum(macro_bleu1) / len(macro_bleu1))
print(sum(macro_bleu2) / len(macro_bleu2))
print(sum(macro_bleu4) / len(macro_bleu4))

print("Distinct")
print(distinct1)
print(distinct2)

print("Speaker Acc")
print(spk_corr / len(y_preds))
