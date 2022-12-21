import torch

import random
import logging
from collections import defaultdict
from itertools import chain
import os
import json

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset

import pdb

logger = logging.getLogger(__file__)

def truncate_batch(batch, padding, align='left', add_length=0):
    trunc = (batch!=padding).int()
    trunc = torch.cat([trunc[:,i:].sum(dim=1,keepdim=True) for i in range(trunc.shape[1])], dim=1)
    trunc = trunc.tolist()
    max_l = max(s.index(0) if 0 in s else len(s) for s in trunc) + add_length
    trunc = [s[:(max_l)] if align=='left' else s[-(max_l):] for s in batch.tolist()]
    return torch.tensor(trunc)

def pad_dataset(dataset, MODEL_INPUTS, padding=0):
    for name in MODEL_INPUTS:
        max_l = max([len(x) for x in dataset[name]])
        if "attention_mask" in name:
            pad_type = 0
        elif "labels" in name:
            pad_type = -100
            max_l += 1
        elif "y_input_ids" in name:
            pad_type = -100
        else:
            pad_type = padding
        dataset[name] = [x + [pad_type] * (max_l - len(x)) for x in dataset[name]]
    return dataset
