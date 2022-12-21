import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

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


class StandardGPT2(nn.Module):
    r"""
    The class wrapping up a GPT2 model with dataloader.

    Args:
        None
    """
    def __init__(self):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-small")
        self.decoder = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-small")
        self.decoder.config.is_encoder_decoder = False
        self.SPECIAL_TOKENS = ["<bos>", "<eos>", "<pad>"]
        self.ATTR_TO_SPECIAL_TOKEN = {
            'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
        }
        self.MODEL_INPUTS = [
            "x_input_ids", "x_attention_mask",
            "y_input_ids", "y_attention_mask",
            "input_ids", "attention_mask", "labels",
        ]
        self.add_special_tokens_()
        self.cache_special_token_ids(self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS))
        self.decoder.config.eos_token_id = self.eos
        self.decoder.config.pad_token_id = self.padding

    def add_special_tokens_(self):
        orig_num_tokens = len(self.tokenizer.encoder)
        num_added_tokens = self.tokenizer.add_special_tokens(
            self.ATTR_TO_SPECIAL_TOKEN)  # returns 0 and doesn't add if they are already there
        if num_added_tokens > 0:
            self.resize_token_embeddings(
                new_num_tokens=orig_num_tokens + num_added_tokens)

    def resize_token_embeddings(self, new_num_tokens=0):
        self.decoder.resize_token_embeddings(new_num_tokens=new_num_tokens)

    def cache_special_token_ids(self, special_token_ids):
        self.special_token_ids = special_token_ids
        self.bos, self.eos, self.padding = special_token_ids

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
        model.resize_token_embeddings(model.decoder.config.vocab_size + len(self.SPECIAL_TOKENS))
        model_state = os.path.join(model_dir, 'pytorch_model.bin')
        state = torch.load(model_state)
        model.load_state_dict(state)
        return model

    def forward(self, batch, labels=None, **generator_args):
        inputs = self._read_batch(batch)
        decoder_inputs = self.form_decoder_inputs(inputs)
        if labels is not None:
            outputs = self.decoder(**decoder_inputs)
        else:
            preds = self.decoder.generate(\
                inputs['x_input_ids'], attention_mask=inputs['x_attention_mask'], **generator_args)
            preds = preds[:,inputs['x_input_ids'].size(-1):]
            trues = inputs['y_input_ids']
            outputs = (preds, trues)
        return outputs


    def form_decoder_inputs(self, ori_inputs):
        inputs = {}
        inputs['input_ids'] = ori_inputs['input_ids']
        inputs['attention_mask'] = ori_inputs['attention_mask']
        inputs['labels'] = ori_inputs['labels']
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
        bos, eos, padding = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS)
        x_sequences = [s for i, s in enumerate(history)]
        y_sequence = [bos] + reply + [eos]

        instance = {}
        instance["x_input_ids"] = list(chain(*x_sequences))[-max_history_len:]
        instance["x_attention_mask"] = [1] * len(instance["x_input_ids"])
        instance["y_input_ids"] = y_sequence
        instance["y_attention_mask"] = [1] * len(instance["y_input_ids"])
        instance["input_ids"] = list(chain(*x_sequences))[-max_history_len:] + y_sequence
        instance["attention_mask"] = [1] * len(instance["input_ids"])
        instance["labels"] = [-100] * (len(list(chain(*x_sequences))[-max_history_len:])+1) + y_sequence[1:]
        return instance


    def get_dataloader(self, data, batch_size, max_history_len=512, shuffle=True, drop_last=False):
        logger.info("Build inputs and labels")
        dataset = defaultdict(list)

        for i, dialog in enumerate(data):
            history = dialog["history"]
            query = dialog["x"]
            label = dialog["y"]
            instance = self.build_input_from_segments(history + [query], label, \
                                                      max_history_len=max_history_len)
            for input_name, input_array in instance.items():
                dataset[input_name].append(input_array)

        logger.info("Pad inputs and convert to Tensor")
        tensor_dataset = []
        dataset = pad_dataset(dataset, self.MODEL_INPUTS, padding=self.padding)
        for input_name in self.MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            tensor_dataset.append(tensor)

        def CustomDataCollator(batch):
            new_batch = []
            for i, input_name in enumerate(self.MODEL_INPUTS):
                reshaped_tensor = torch.cat([b[i].view(1,-1) for b in batch],dim=0)
                if "y_input_ids" in input_name:
                    new_batch.append(truncate_batch(reshaped_tensor, -100))
                elif "input_ids" in input_name:
                    new_batch.append(truncate_batch(reshaped_tensor, self.padding))
                elif "attention_mask" in input_name:
                    new_batch.append(truncate_batch(reshaped_tensor, 0))
                elif "labels" in input_name:
                    new_batch.append(truncate_batch(reshaped_tensor, -100))
            return tuple(new_batch)

        logger.info("Build dataloader")
        dataset = TensorDataset(*tensor_dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, \
                                collate_fn=CustomDataCollator, pin_memory=True)

        logger.info("set x (Batch, Seq length): {}".format(dataset.tensors[0].shape))
        logger.info("set y (Batch, Seq length): {}".format(dataset.tensors[2].shape))
        logger.info("set input_ids (Batch, Seq length): {}".format(dataset.tensors[4].shape))
        return dataloader
