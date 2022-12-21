# CausalDialogue
This repository contains code and data used in this [paper](https://arxiv.org/pdf/2212.10515.pdf).
The code part is under MIT License; and the data part, as described in `data/README.md` is under GNU Free Document License.
We have included the license documents in this repository.
If you find the code or data is useful for your research, please kindly cite the paper.

## Setup
* clone this repository, you might also need to refer to [Git LFS](https://git-lfs.com/) in order to download the data.
* (recommend) create a conda environment.
* install packages via conda and pip install.
```
$conda install graphviz
$pip install -r requirements.txt
```
* install required nltk package
```
$python
>>> import nltk
>>> nltk.download('punkt')
```

## Usage

* Training; an example of fine-tuning a T5 model on CausalDialog using MLE
```
python main.py --task fethco --model_type t5 --lr 1e-5 --n_epochs 5 --train_batch_size 16 --valid_batch_size 32 --gradient_accumulation_steps 4 --max_history_len 256 --model_checkpoint exp_t5_standard --do_train
```
* Testing; an example of generation on test set using TopK sampling
```
python main.py --task fethco --model_type t5 --model_checkpoint exp_t5_standard/pytorch_model.bin --max_history_len 256 --do_test --do_generate --loss_type standard --sample_method topk --test_batch_size 128 --preds_outpath exp_t5_standard/ckpt_last_topk
```
* To get automatic evaluation results, we currently follow three steps.
    * first utilize `visualize_gt_data.py` to have the references.
    * second utilize `gt_eval.py` to get some reference of the ground-truths results.
    * third utilize `post_eval.py` to get the results of the generated file in the testing stage.
* Some notes:
    * `--task fethco` is an alias and required for the `main.py`. For now, there is no other tasks written in the program.
    * `--model_type` can be set to `t5` or `dialogpt` for now.
    * `--loss_type` can be set to `standard` or `exmate` for now.
    * `--sample_method` can be set to `argmax`, `softmax`, or `topk` for now.
