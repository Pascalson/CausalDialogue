import torch
import sys
import tqdm
import transformers
from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-base")
data = torch.load("caches/cache_fethco_test_dag_T5Tokenizer")

with open("test_v1_dialog_history.txt","w") as fout1, open("test_v1_gt_response.txt","w") as fout2:
    for datum in tqdm.tqdm(data):
        tmp_context = datum['history'] + [datum['x']]
        context = ""
        for utt in tmp_context:
            context += tokenizer.decode(utt) + " | "
        response = tokenizer.decode(datum['y'])
        fout1.write(context+"\n")
        fout2.write(response+"\n")
