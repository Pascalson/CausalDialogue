"""
evaluate PPL, ConfCE of ground-truths
PPL: include speaker prediction
"""
from collections import defaultdict
from transformers import T5Tokenizer
import numpy as np
import pdb

tokenizer = T5Tokenizer.from_pretrained("t5-base")

with open('test_v1_dialog_history.txt','r') as fin:
    DHs = fin.readlines()

with open('test_v1_gt_response.txt','r') as fin:
    y_trues = fin.readlines()

tokenized_y_trues = []
multi_y_trues = defaultdict(list)
fork_yts = defaultdict(list)
for DH, yt in zip(DHs, y_trues):
    tokenized_yt = tokenizer.tokenize(yt.strip())
    multi_y_trues[DH].append(tokenized_yt)
    tokenized_y_trues.append(tokenized_yt)

    try:
        only_DH, x = DH.strip()[:-1].rsplit('|',1)
    except:
        x = DH.strip()[:-1]
        only_DH = x + 'root'
    fork_yts[only_DH].append((DH, x, tokenized_yt))


# Calculate Cross-Entropy Loss and Perplexity
def calc_ppl(DHs, yts):
    loss = 0.
    total_tokens = 0
    for DH, yt in zip(DHs, yts):
        yt_count = sum(yt == y for y in multi_y_trues[DH])
        logp_yt = - np.log(yt_count / len(multi_y_trues[DH]) + 1e-7)
        loss += logp_yt * len(yt)#assume each token is the same probability as the utterance, this approximiation can be over-pessimistic
        total_tokens += len(yt)
    loss /= total_tokens
    print(loss)
    print(total_tokens)
    print(np.exp(loss))

calc_ppl(DHs, tokenized_y_trues)


# Calculate ConfCE
pos_pairs, neg_pairs = [], []
for key, value in fork_yts.items():
    xy_pairs = [(v[1],v[2]) for v in value]
    DHs = [v[0] for v in value]
    xs = [v[1] for v in value]
    ys = [v[2] for v in value]

    count = 0
    for DH, x in zip(DHs, xs):
        for y in ys:
            if (DH,x,y) not in value:
                count += 1
                if (DH,y) not in neg_pairs:
                    neg_pairs.append((DH,y))
    if count > 0:
        for (DH,x,y) in value:
            if (DH,y) not in pos_pairs:
                pos_pairs.append((DH,y))

print("Neg Pairs")
print(len(neg_pairs))
calc_ppl([v[0] for v in neg_pairs], [v[1] for v in neg_pairs])

print("Pos Pairs")
print(len(pos_pairs))
calc_ppl([v[0] for v in pos_pairs], [v[1] for v in pos_pairs])
