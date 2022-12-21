import torch
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from ignite.metrics.nlp import Bleu

import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

from nltk import ngrams
from nltk.tokenize import word_tokenize
from itertools import chain


class CustomDialogPostMetrics(Metric):
    r"""
    customed Dialog Post-processed Metrics
    note that in current version, we set the self.only_generate = True
    """

    def __init__(self, model_type, tokenizer, eos_id, output_transform=lambda x: x, speaker_separator=None, device="cpu", prediction_outpath=None):
        self.only_generate = True# to turn off all metrics but only generate results, this can facilitate the speed and do not need to regenerate every time
        self.model_type = model_type

        self.prediction_fout = None
        if prediction_outpath is not None:
            self.prediction_fout = open(prediction_outpath,"w")

        self.tokenizer = tokenizer
        self.eos_id = eos_id

        if not self.only_generate:
            self.speaker_separator = None
            if speaker_separator is not None:
                self.speaker_separator = speaker_separator
            self.bleu1_m = Bleu(ngram=1, smooth="smooth1")
            self.bleu2_m = Bleu(ngram=2, smooth="smooth1")
            self._DH_ids = []
            self._data_ids = []
            self._xs = []
        self._y_preds = []
        self._y_trues = []
        super(CustomDialogPostMetrics, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        if not self.only_generate:
            self._DH_ids = []
            self._data_ids = []
            self._xs = []
            self.bleu1_m.reset()
            self.bleu2_m.reset()
        self._y_preds = []
        self._y_trues = []
        super(CustomDialogPostMetrics, self).reset()

    @reinit__is_reduced
    def update(self, output):

        if self.prediction_fout is not None or not self.only_generate:
            y_pred, y_true, DH_ids, xs, data_ids = output[0].detach(), output[1].detach(), output[2].detach(), output[3].detach(), output[4].detach()

        if not self.only_generate:
            xs = [x[:x.index(self.tokenizer.pad_token_id)] if self.tokenizer.pad_token_id in x else x for x in xs.tolist()]
            y_true = [y[:y.index(-100)] if -100 in y else y for y in y_true.tolist()]#remove id=-100, which is for neglection in labels
        if self.prediction_fout is not None or not self.only_generate:
            y_preds = [y[:y.index(self.eos_id)] if self.eos_id in y else y for y in y_pred.tolist()]

        # transform to natural language
        if not self.only_generate:
            xs = self.tokenizer.batch_decode(xs, skip_special_tokens=True)
            y_trues = self.tokenizer.batch_decode(y_true, skip_special_tokens=True)
        if self.prediction_fout is not None or not self.only_generate:
            y_preds = self.tokenizer.batch_decode(y_pred, skip_special_tokens=True)

        # save predictions to a file, if specified
        if self.prediction_fout is not None:
            for pred in y_preds:
                self.prediction_fout.write(pred+"\n")

        if not self.only_generate:
            # separate speaker if self.speaker_separator is specified
            if self.speaker_separator is not None:
                spk_trues, spk_preds = [],[]
                pure_y_trues, pure_y_preds = [],[]
                for yt,yp in zip(y_trues, y_preds):

                    if self.speaker_separator in yt:
                        spk_t, pure_yt = yt.split(self.speaker_separator,1)
                    else:
                        spk_t = ''
                        pure_yt = yt

                    if self.speaker_separator in yp:
                        spk_p, pure_yp = yp.split(self.speaker_separator,1)
                    else:
                        spk_p = ''
                        pure_yp = yp
                    spk_trues.append(spk_t.strip())
                    spk_preds.append(spk_p.strip())
                    pure_y_trues.append(pure_yt.strip())
                    pure_y_preds.append(pure_yp.strip())
                y_trues = pure_y_trues
                y_preds = pure_y_preds

            # tokenized by NLTK package as unification
            y_preds = [word_tokenize(y) for y in y_preds]
            y_trues = [word_tokenize(y) for y in y_trues]
            self._DH_ids.extend(DH_ids.squeeze().tolist())
            self._data_ids.extend(data_ids.squeeze().tolist())
            self._xs.extend(xs)
            self._y_preds.extend(y_preds)
            self._y_trues.extend(y_trues)


    @sync_all_reduce("_y_preds", "_y_trues")
    def compute(self):
        if self.prediction_fout is not None:
            self.prediction_fout.close()
        if self.only_generate:
            return {}

        # if do normal post generation metrics
        # for BLEU with multiple reference
        multi_y_trues = defaultdict(list)
        for DH_id, x, y in zip(self._DH_ids, self._xs, self._y_trues):
            multi_y_trues[(DH_id, x)].append(y)
        bleu_refs = []
        for DH_id, x in zip(self._DH_ids, self._xs):
            bleu_refs.append(multi_y_trues[(DH_id, x)])
        self.bleu1_m.update((self._y_preds, bleu_refs))
        self.bleu2_m.update((self._y_preds, bleu_refs))

        # for distinct-N
        ngrams1 = list(chain(*[[gram for gram in ngrams(y, 1)] for y in self._y_preds]))
        ngrams2 = list(chain(*[[gram for gram in ngrams(y, 2)] for y in self._y_preds]))
        gt_ngrams1 = list(chain(*[[gram for gram in ngrams(y, 1)] for y in self._y_trues]))
        gt_ngrams2 = list(chain(*[[gram for gram in ngrams(y, 2)] for y in self._y_trues]))

        return {
            'bleu1': self.bleu1_m.compute().item(), \
            'bleu2': self.bleu2_m.compute().item(), \
            'distinct-1': len(set(ngrams1)) / len(ngrams1), \
            'distinct-2': len(set(ngrams2)) / len(ngrams2), \
            'gt-distinct-1': len(set(gt_ngrams1)) / len(gt_ngrams1), \
            'gt-distinct-2': len(set(gt_ngrams2)) / len(gt_ngrams2), \
        }




class CustomAvgLoss(Metric):
    r"""
    customed Average Loss to reduce the effect of different batchsize.
    """

    def __init__(self, output_transform=lambda x: x, device="cpu"):
        self._losses = []
        self.totalsize = 0
        super(CustomAvgLoss, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._losses = []
        self.totalsize = 0
        super(CustomAvgLoss, self).reset()

    @reinit__is_reduced
    def update(self, output):
        loss, batchsize = output[0], output[1]
        self._losses.append(loss*batchsize)
        self.totalsize += batchsize

    @sync_all_reduce("_losses","totalsize")
    def compute(self):
        return sum(self._losses) / self.totalsize
