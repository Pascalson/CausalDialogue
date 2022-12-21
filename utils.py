from datetime import datetime
import json
import logging
import os
import math
import tqdm
from collections import defaultdict
import random
import string

import torch
from transformers import cached_path

from dialdag import DialDAG

import pdb

logger = logging.getLogger(__file__)
def get_dataset(tokenizer, dataset_path, dataset_cache=None, lower_case=False, no_punctuation=False):
    """Tokenize and encode dataset and save/read dataset from cache"""
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized and encoded dataset at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Tokenize and encode dataset %s", dataset_path)
        data_file = cached_path(dataset_path)
        with open(data_file, "r", encoding="utf-8") as f:
            dataset_json = json.loads(f.read())

        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            return list(tokenize(o) for o in obj)

        dataset = []
        for i in tqdm.tqdm(range(len(dataset_json))):
            # save into DAG format
            if len(dataset_json[i]["original_data"]) > 100:
                continue
            dialog_dag = DialDAG(dataset_json[i]["original_data"])
            if dataset_json[i]['category'] == 'original scripts':
                if 'Explor' in dataset_json[i]['dialog_name']:#include Explore and Exploration
                    continue
                if "The_Cause_of_Sorrow_Script.json__The_White_Heron_Cup" in dataset_json[i]['dialog_name']:
                    continue

            # find all possible dialogue branches
            dialog_paths = dialog_dag.get_possible_paths(expansion_rate=0.0)

            # save into dataset
            used_subpaths = [[]]
            for k, dialog_path in enumerate(dialog_paths):
                for j, node_id in enumerate(dialog_path[1:]):
                    
                    if dialog_dag.nodes[node_id].get_type() != "utterance":
                        continue
                    if dialog_dag.nodes[node_id].get_utt() == "...":#added to filter out responses that do not need to learn
                        continue

                    # avoid repeating the exactly same dialogue
                    subpath = dialog_path[:j+2]
                    if subpath in used_subpaths:
                        continue
                    else:
                        used_subpaths.append(subpath)

                    # find DH_id
                    if dialog_path[:j] not in used_subpaths:
                        used_subpaths.append(dialog_path[:j])
                    local_DH_id = used_subpaths.index(dialog_path[:j])
                    DH_id = i*100 + local_DH_id# allow a tolerance of 100 maximum utts in a path

                    # format the data point
                    data_id = i*10000 + k*100 + j# all a tolerance of 100 maximum possible paths and each path has 100 maximum utts
                    history = [dialog_dag.nodes[nid].get_speaker() + ": " + dialog_dag.nodes[nid].get_utt() for nid in dialog_path[:j]]
                    x = dialog_dag.nodes[dialog_path[j]].get_speaker() + ": " \
                        + dialog_dag.nodes[dialog_path[j]].get_utt()
                    y = dialog_dag.nodes[node_id].get_speaker() + ": " \
                        + dialog_dag.nodes[node_id].get_utt()

                    if lower_case:
                        history = [utt.lower() for utt in history]
                        x = x.lower()
                        y = y.lower()

                    if no_punctuation:
                        history = [utt.split(':',1)[0] + ':' + utt.split(':',1)[1].translate(str.maketrans('', '', string.punctuation)) for utt in history]
                        x = x.split(':',1)[0] + ':' + x.split(':',1)[1].translate(str.maketrans('', '', string.punctuation))
                        y = y.split(':',1)[0] + ':' + y.split(':',1)[1].translate(str.maketrans('', '', string.punctuation))

                    dataset.append({
                        'data_id': data_id,
                        'DH_id': DH_id,
                        'history':tokenize(history),
                        'x':tokenize(x),
                        'y':tokenize(y),
                    })

        if dataset_cache:
            torch.save(dataset, dataset_cache)

    # collect the DH_id dictionary
    DH_id_dict = defaultdict(list)
    for d in dataset:
        DH_id_dict[d['DH_id']].append(d['data_id'])

    return dataset, DH_id_dict


def create_fake_causal_effects(dataset, DH_id_dict):
    datadict_by_id = {}
    for d in dataset:
        datadict_by_id[d['data_id']] = d

    fake_ce = []
    pos_ce = []
    for DH_id, data_ids in DH_id_dict.items():
        if len(data_ids) > 1:
            xy_pairs = [(datadict_by_id[data_id]['x'],datadict_by_id[data_id]['y']) for data_id in data_ids]
            xs = [item[0] for item in xy_pairs]
            ys = [item[1] for item in xy_pairs]
            count = 0
            for x in xs:
                for y in ys:
                    if (x,y) not in xy_pairs:
                        count += 1
                        fake_ce.append({
                            'data_id':-1,
                            'DH_id':DH_id,
                            'history':datadict_by_id[data_ids[0]]['history'],
                            'x':x,
                            'y':y,
                        })
            if count > 0:
                for (x,y) in xy_pairs:
                    pos_ce.append({
                        'data_id':-2,
                        'DH_id':DH_id,
                        'history':datadict_by_id[data_ids[0]]['history'],
                        'x':x,
                        'y':y,
                    })

    return fake_ce, pos_ce

def make_logdir(model_name: str):
    """Create unique path to save results and checkpoints, e.g. runs/Sep22_19-45-59_gpu-7_gpt2"""
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = model_name + '_' + current_time
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    return logdir
