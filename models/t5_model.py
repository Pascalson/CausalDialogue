import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration

import os
import json
import random
import logging
from collections import defaultdict
from itertools import chain
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from models.utils import truncate_batch, pad_dataset

import pdb
logger = logging.getLogger(__file__)

class StandardT5(nn.Module):
    r"""
    The class wrapping up a T5 model with dataloader.

    Args:
        None
    """
    def __init__(self):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.s2s_model = T5ForConditionalGeneration.from_pretrained("t5-base")
        self.MODEL_INPUTS = [
            "x_input_ids", "x_attention_mask",
            "y_input_ids", "y_attention_mask", "labels",
        ]
        self.padding = self.tokenizer.pad_token_id
        self.cache_special_token_ids([self.padding, self.tokenizer.eos_token_id])


    def cache_special_token_ids(self, special_token_ids):
        self.special_token_ids = special_token_ids
        if isinstance(special_token_ids, list):
            self.padding, self.eos = special_token_ids
        else:
            self.padding = special_token_ids

    def _read_batch(self, batch):
        inputs = {}
        for name, name_batch in zip(self.MODEL_INPUTS, batch):
            inputs[name] = name_batch
        return inputs

    @classmethod
    def from_pretrained(cls, model_dir):
        args_file = os.path.join(model_dir, 'model_training_args.bin')
        args = torch.load(args_file)
        model = cls(args)
        model_state = os.path.join(model_dir, 'pytorch_model.bin')
        state = torch.load(model_state)
        model.load_state_dict(state)
        return model

    def forward(self, batch, labels=None, ignore_separator_id=None, **generator_args):
        inputs = self._read_batch(batch)
        decoder_inputs = self.form_model_inputs(inputs)
        if labels is not None:
            outputs = self.s2s_model(**decoder_inputs)
            if ignore_separator_id is not None:
                # replace tokens before the ignore_separator_id to -100, the ignore_index for cross_entropy
                where_to_ignore = (decoder_inputs['labels'] == ignore_separator_id)
                decoder_inputs['labels'][where_to_ignore] = -100
                for t in range(decoder_inputs['labels'].size(-1)-1):
                    where_to_ignore[:,t+1] = torch.logical_or(where_to_ignore[:,t], where_to_ignore[:,t+1])
                where_to_ignore = torch.logical_not(where_to_ignore)
                decoder_inputs['labels'][where_to_ignore] = -100
                # re-compute the cross-entropy loss
                outputs.loss = F.cross_entropy(outputs.logits.view(-1,outputs.logits.size(-1)), decoder_inputs['labels'].view(-1))
        else:
            preds = self.s2s_model.generate(\
                inputs['x_input_ids'], attention_mask=inputs['x_attention_mask'], **generator_args)
            trues = inputs['y_input_ids']
            outputs = (preds, trues)
        return outputs


    def form_model_inputs(self, ori_inputs):
        r"""
        The function reset the keys of the input dict.

        Args:
            ori_inputs (`dictionary`):
                The input dictionary for changing the names.
        """
        inputs = {}
        inputs['input_ids'] = ori_inputs['x_input_ids']
        inputs['attention_mask'] = ori_inputs['x_attention_mask']
        inputs['labels'] = ori_inputs['y_input_ids']
        return inputs


    def build_input_from_segments(self, history, reply, max_history_len=512):
        r"""
        The function to concatenate dialogue history as encoder inputs
        and add eos token to the response as the decoder inputs.

        Args:
            history: (`list`)
                The turns in dialogue history.
            reply: (`list`)
                The tokens of the response.
            max_history_len: (`int`)
                The maximum allowed length of the dialogue history.
        """
        x_sequences = [s for i, s in enumerate(history)]
        y_sequence = reply + [self.tokenizer.eos_token_id]
        instance = {}
        instance["x_input_ids"] = list(chain(*x_sequences))[-max_history_len:]
        instance["x_attention_mask"] = [1] * len(instance["x_input_ids"])
        instance["y_input_ids"] = y_sequence
        instance["y_attention_mask"] = [1] * len(instance["y_input_ids"])
        instance["labels"] = y_sequence[1:]
        return instance


    def get_dataloader(self, data, batch_size, max_history_len=512, shuffle=True, drop_last=False):
        logger.info("Build inputs and labels")
        dataset = defaultdict(list)

        for i, dialog in enumerate(data):
            history    = dialog["history"]
            last_utt   = dialog["x"]
            response   = dialog["y"]
            instance = self.build_input_from_segments(history + [last_utt], response, \
                                                      max_history_len=max_history_len)
            for input_name, input_array in instance.items():
                dataset[input_name].append(input_array)
            dataset['data_id'].append(dialog['data_id'])
            dataset['DH_id'].append(dialog['DH_id'])

        logger.info("Pad inputs and convert to Tensor")
        tensor_dataset = []
        dataset = pad_dataset(dataset, self.MODEL_INPUTS, padding=self.padding)
        for input_name in self.MODEL_INPUTS + ['data_id', 'DH_id']:
            tensor = torch.tensor(dataset[input_name])
            tensor_dataset.append(tensor)

        def CustomDataCollator(batch):
            new_batch = []
            for i, input_name in enumerate(self.MODEL_INPUTS + ['data_id', 'DH_id']):
                reshaped_tensor = torch.cat([b[i].view(1,-1) for b in batch],dim=0)
                if "x_input_ids" in input_name:
                    new_batch.append(truncate_batch(reshaped_tensor, self.padding))
                elif "y_input_ids" in input_name:
                    new_batch.append(truncate_batch(reshaped_tensor, -100))
                elif "attention_mask" in input_name:
                    new_batch.append(truncate_batch(reshaped_tensor, 0))
                elif "labels" in input_name:
                    new_batch.append(truncate_batch(reshaped_tensor, -100, add_length=1))
                elif input_name in ['data_id', 'DH_id']:
                    new_batch.append(reshaped_tensor)
            return tuple(new_batch)

        logger.info("Build dataloader")
        dataset = TensorDataset(*tensor_dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, \
                                collate_fn=CustomDataCollator, pin_memory=True)

        logger.info("set x (Batch, Seq length): {}".format(dataset.tensors[0].shape))#x_input_ids
        logger.info("set y (Batch, Seq length): {}".format(dataset.tensors[2].shape))#y_input_ids
        return dataloader
