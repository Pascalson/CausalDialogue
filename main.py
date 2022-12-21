import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage, Average
from ignite.metrics.nlp import Bleu
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear

from utils import make_logdir, get_dataset, create_fake_causal_effects#, create_DCE_pair_ids
from transformers import (AdamW, WEIGHTS_NAME, CONFIG_NAME)
from models.gpt2_model import StandardGPT2
from models.t5_model import StandardT5
from metrics import CustomDialogPostMetrics, CustomAvgLoss#, CustomDCEMetrics

import os
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
import pdb

logger = logging.getLogger(__file__)
torch.autograd.set_detect_anomaly(True)

MODELS = ["t5","dialogpt"]


def set_args():
    r"""
    The function to set arguments.

    Args:
        None
    """
    parser = ArgumentParser()
    
    # Set the task
    parser.add_argument("--task", type=str, required=True,
        help="The task name")

    # Data preprocessing
    parser.add_argument("--lower_case", action='store_true')
    parser.add_argument("--no_punctuation", action='store_true')

    # Setup the model
    parser.add_argument("--model_type", type=str, default="t5", 
        help="Short name of the model. Can be chose from \"dialogpt\" or \"t5\" now.")
    parser.add_argument("--model_checkpoint", type=str, required=True,
        help="Path to save the model")
    parser.add_argument("--preds_outpath", type=str, default="tmp.txt",
        help="Path to save generation results")
    parser.add_argument("--loss_type", type=str, default="standard",
        help="Used training loss type: 'standard', 'mate', 'exmate'.")
    parser.add_argument("--mate_coef", type=float, default=0.05,
        help="mate loss coefficient")
    
    # Setup hyperparameters
    parser.add_argument("--max_history_len", type=int, default=256,
        help="Maximum length of dialogue history taken as input")
    parser.add_argument("--train_batch_size", type=int, default=4,
        help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4,
        help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=4,
        help="Batch size for testing")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
        help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=1e-5,
        help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0,
        help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=5,
        help="Number of training epochs")
    parser.add_argument("--schedule_lr", action='store_true',
        help="If true use learning rate scheduler")
    
    # others
    parser.add_argument('--do_train', action="store_true")
    parser.add_argument('--do_valid', action="store_true")
    parser.add_argument('--do_test', action="store_true")
    parser.add_argument('--do_confce_test', action="store_true")
    parser.add_argument('--do_generate', action="store_true")
    parser.add_argument('--sample_method', type=str, default="argmax")
    parser.add_argument("--eval_before_start", action='store_true', 
        help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)")
    
    args = parser.parse_args()

    return args


def run():
    r"""
    This is the main script to train and evaluate dialogue response generation.

    Args:
        None
    """
    # Read arguments and set the logger
    args = set_args()
    logging.basicConfig(level=logging.INFO)
    logger.info("Arguments: %s", pformat(args))
    logger.info("Prepare models")
    assert args.model_type in MODELS, "args.model_type is not in acceptable MODELS list {}".format(MODELS)
    
    # Setup task, model, tokenizer, dataloaders, and optimizer
    if args.task == "fethco":
        train_set_path  = "data/train_dag.json"
        valid_set_path  = "data/valid_dag.json"
        test_set_path   = "data/test_dag.json"
        if not os.path.isdir("caches"):
            os.makedirs("caches")
        train_set_cache = "caches/cache_fethco_train_dag"
        valid_set_cache = "caches/cache_fethco_valid_dag"
        test_set_cache  = "caches/cache_fethco_test_dag"
        if args.lower_case:
            train_set_cache += '_lowercase'
            valid_set_cache += '_lowercase'
            test_set_cache += '_lowercase'
        if args.no_punctuation:
            train_set_cache += '_nopunctuation'
            valid_set_cache += '_nopunctuation'
            test_set_cache += '_nopunctuation'


    if args.model_type == "t5":
        model = StandardT5().to(args.device)
    elif args.model_type in "dialogpt":
        model = StandardGPT2().to(args.device)
    tokenizer = model.tokenizer

    if os.path.exists(args.model_checkpoint):
        print("{} exists, loading the state dict".format(args.model_checkpoint))
        state = torch.load(args.model_checkpoint)
        model.load_state_dict(state)

    if args.do_train:
        train_set, train_DH_id_dict = get_dataset(tokenizer, train_set_path, train_set_cache, lower_case=args.lower_case, no_punctuation=args.no_punctuation)
        train_loader = model.get_dataloader(train_set, args.train_batch_size, max_history_len = args.max_history_len)
        neg_train_set, pos_train_set = create_fake_causal_effects(train_set, train_DH_id_dict)
        neg_train_loader = model.get_dataloader(neg_train_set, args.train_batch_size, max_history_len = args.max_history_len)
        neg_train_loader_iter = iter(neg_train_loader)
    if args.do_train or args.do_valid:
        valid_set, valid_DH_id_dict = get_dataset(tokenizer, valid_set_path, valid_set_cache, lower_case=args.lower_case, no_punctuation=args.no_punctuation)
        valid_loader = model.get_dataloader(valid_set, args.valid_batch_size, max_history_len = args.max_history_len, shuffle=False)
    if args.do_test:
        test_set, test_DH_id_dict = get_dataset(tokenizer, test_set_path, test_set_cache, lower_case=args.lower_case, no_punctuation=args.no_punctuation)
        test_loader = model.get_dataloader(test_set, args.test_batch_size, max_history_len = args.max_history_len, shuffle=False)
    if args.do_confce_test:
        test_set, test_DH_id_dict = get_dataset(tokenizer, test_set_path, test_set_cache, lower_case=args.lower_case, no_punctuation=args.no_punctuation)
        neg_test_set, pos_test_set = create_fake_causal_effects(test_set, test_DH_id_dict)
        neg_test_loader = model.get_dataloader(neg_test_set, args.test_batch_size, max_history_len = args.max_history_len, shuffle=False)
        pos_test_loader = model.get_dataloader(pos_test_set, args.test_batch_size, max_history_len = args.max_history_len, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)


    def update(engine, batch):
        r"""
        The function to update model.

        Args:
            engine (`ignite.engine.Engine)`:
                A training framework that will call this function by `Engine(update)` every iteration.
            batch ():
                One batch of data from the dataloader that is passed by `Engine(update)` to this function at every iteration.
        """
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)

        outputs = model(batch, labels=batch)
        loss = outputs.loss
        if args.loss_type in ["mate","exmate"]:
            try:
                untreated_batch = next(neg_train_loader_iter)
            except:
                neg_train_loader_iter = iter(neg_train_loader)
                untreated_batch = next(neg_train_loader_iter)
            untreated_batch = tuple(input_tensor.to(args.device) for input_tensor in untreated_batch)
            untreated_outputs = model(untreated_batch, labels = untreated_batch, ignore_separator_id = tokenizer.convert_tokens_to_ids(':'))
            untreated_loss = - untreated_outputs.loss if args.loss_type == "mate" else torch.exp(-untreated_outputs.loss)
            loss = loss + args.mate_coef*untreated_loss

        loss /= args.gradient_accumulation_steps
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        return (loss.item() * args.gradient_accumulation_steps,)


    def inference(engine, batch):
        r"""
        The function to evaluate model.

        Args:
            engine (`ignite.engine.Engine)`:
                A training framework that will call this function by `Engine(update)` every iteration.
            batch ():
                One batch of data from the dataloader that is passed by `Engine(update)` to this function at every iteration.
        """
        model.eval()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        # batch[-2]: dialog_id
        # batch[-1]: DH_id
        with torch.no_grad():
            outputs = model(batch, labels=batch)
            loss = outputs.loss# only look into perplexity (not mate) in validation stage
            # use outputs.logits
            infer_preset_len = 0
            if args.model_type == 'dialogpt':
                infer_preset_len = batch[0].shape[-1]# the length of dialogue history in this batch
            if args.sample_method == "argmax":
                preds, trues = model(batch, min_length=10+infer_preset_len, max_length=20+infer_preset_len)
            elif args.sample_method == "softmax":
                preds, trues = model(batch, min_length=10+infer_preset_len, max_length=50+infer_preset_len, temperature=0.5, do_sample=True)
            elif args.sample_method == "topk":
                preds, trues = model(batch, min_length=3+infer_preset_len, max_length=50+infer_preset_len, top_k=10, do_sample=True)
        return {'loss':loss.item(), 'tokensize':(batch[2]!=-100).sum().item(), 'preds':preds, 'trues':trues, 'DH_id':batch[-1], 'xs':batch[0], 'data_id':batch[-2]}


    # Setup ignite engines and results tracker
    pbar = ProgressBar(persist=True)

    evaluator = Engine(inference)

    if args.do_generate:
        CustomDialogPostMetrics(model_type=args.model_type, tokenizer=tokenizer, eos_id=model.eos, speaker_separator=':', prediction_outpath=args.preds_outpath+'.txt' if args.do_generate else None, output_transform=lambda x: (x['preds'], x['trues'], x['DH_id'], x['xs'], x['data_id'])).attach(evaluator, 'DPM')
    CustomAvgLoss(output_transform=lambda x: (x['loss'],x['tokensize'])).attach(evaluator, 'loss')

    pbar.attach(evaluator)
    evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat({m:v for m,v in evaluator.state.metrics.items() if m not in ['DPM']})))

    if args.do_train:
        trainer = Engine(update)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(valid_loader))
        if args.n_epochs < 1:
            trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(valid_loader))
        if args.eval_before_start:
            trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(valid_loader))

        if args.schedule_lr:
            scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
            trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

        RunningAverage(output_transform=lambda x: x[0]).attach(trainer, "loss")
        pbar.attach(trainer, metric_names=["loss"])


        # Save checkpoints
        log_dir = make_logdir(args.model_checkpoint)
        checkpoint_handler = ModelCheckpoint(log_dir, 'checkpoint', save_interval=1, n_saved=args.n_epochs)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})
        torch.save(args, log_dir + '/model_training_args.bin')
        tokenizer.save_pretrained(log_dir)

        # Save losses
        train_loss_logger = []
        valid_loss_logger = []

        def save_losses():
            torch.save({'train':train_loss_logger,'valid':valid_loss_logger}, \
                os.path.join(log_dir, 'losses.bin'))

        @trainer.on(Events.EPOCH_COMPLETED)
        def save_trainer_logger(engine):
            train_loss_logger.append(engine.state.metrics.copy())
            save_losses()

        @evaluator.on(Events.COMPLETED)
        def save_valid_logger(engine):
            valid_loss_logger.append(engine.state.metrics.copy())
            save_losses()

    if args.do_train:
        trainer.run(train_loader, max_epochs=args.n_epochs)
        # Rename the last checkpoint with a default name to be easier loaded by "from_pretrained()"
        if args.n_epochs > 0:
            os.rename(checkpoint_handler.last_checkpoint, os.path.join(log_dir, WEIGHTS_NAME))
    elif args.do_valid:
        evaluator.run(valid_loader, max_epochs=1)
    elif args.do_test or args.do_diffbleu_test:
        evaluator.run(test_loader, max_epochs=1)
    elif args.do_confce_test:
        evaluator.run(neg_test_loader, max_epochs=1)
        evaluator.run(pos_test_loader, max_epochs=1)

if __name__ == "__main__":
    run()
