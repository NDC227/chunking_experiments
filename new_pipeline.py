import argparse
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
from datasets import Dataset, get_dataset_config_names
import wandb
import torch.cuda
import lightning as L
import numpy as np

from replug_transformer_new import ReplugTransformer

# Sample use
# python new_pipeline.py --llm_name facebook/opt-125m --train --tiny --dataset toy_rechunked_and_scored_nq --train_epochs 2 --batch_size 2
# python new_pipeline.py --llm_name facebook/opt-125m --eval --eval_experiment 1 --tiny --batch_size 2 --dataset new_chunks_with_retrieve --eval_k 10
# python new_pipeline.py --llm_name facebook/opt-125m --eval --eval_experiment 1.5 --tiny --batch_size 2 --dataset new_chunks_with_retrieve --eval_k 10
# python new_pipeline.py --llm_name facebook/opt-125m --eval --eval_experiment 2 --tiny --batch_size 1 --dataset test_rechunked --exp2_subset RecursiveCharacterTextSplitter_250_0 --eval_k 10
# python new_pipeline.py --llm_name facebook/opt-125m --eval --eval_experiment 3 --tiny --batch_size 1 --dataset test_rechunked --eval_k 10
# python new_pipeline.py --llm_name facebook/opt-125m --eval --eval_experiment 3.5 --tiny --batch_size 2 --dataset toy_rechunked_and_scored_nq --eval_k 10
argp = argparse.ArgumentParser()
argp.add_argument('--llm_name', default='facebook/opt-125m')
argp.add_argument('--train', action='store_true')
argp.add_argument('--eval', action='store_true')
argp.add_argument('--eval_experiment', default='1')
argp.add_argument('--exp2_subset', default='')
argp.add_argument('--tiny', action='store_true')
argp.add_argument('--save_name', default=None)
argp.add_argument('--baseline', action='store_true')
argp.add_argument('--batch_size', default=8, type=int)
argp.add_argument('--train_epochs', default=1, type=int)
argp.add_argument('--lr', default=1e-4, type=float)
argp.add_argument('--eval_k', default=10, type=int)
argp.add_argument('--train_set', default='toy_rechunked_and_scored_nq')
argp.add_argument('--valid_set', default='valid_chunks_with_retrieve')
argp.add_argument('--dataset', default='toy_rechunked_and_scored_nq')
argp.add_argument('--num_proc', default=1, type=int)
argp.add_argument('--num_gpus', default=0, type=int)
argp.add_argument('--add_ids', action='store_true')
argp.add_argument('--length_penalty', default=0.0, type=float)
argp.add_argument('--eval_reranker', default=None)
argp.add_argument('--debug', action='store_true')
args = argp.parse_args()

os.environ['WANDB_INIT_TIMEOUT'] = '300'
os.environ['HF_TOKEN'] = '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'
torch.set_float32_matmul_precision('medium')
torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_id = args.llm_name
num_proc = args.num_proc
batch_size = args.batch_size

def print_debug(*prints):
    if args.debug:
        print(prints)

def sample_docs(ex):
    idxs = torch.randint(len(ex['new_chunks']), (5,))
    ex['new_chunks'] = np.take(ex['new_chunks'],idxs)
    ex['chunker_ids'] = np.take(ex['chunker_ids'],idxs)
    ex['llm_scores'] = np.take(ex['llm_scores'],idxs)
    return ex

def downsample_eval(ex):
    idxs = torch.randint(len(ex['new_chunks']), (100,))
    ex['new_chunks'] = np.take(ex['new_chunks'],idxs)
    ex['chunker_ids'] = np.take(ex['chunker_ids'],idxs)
    return ex

def combine_subsets(ex, idx):
    for i in range(1, len(subsets)):
        ex['new_chunks'].extend(subsets[i][idx]['new_chunks'])
        ex['chunker_ids'].extend(subsets[i][idx]['chunker_ids'])
    return ex

model = ReplugTransformer(model_id)

if args.train:
    if args.tiny:
        train_dataset = load_dataset(f'ndc227/{args.dataset}', split='train', streaming=True, cache_dir='/nlp/scr/ayc227/.cache/huggingface/datasets').take(10)
        train_dataset = Dataset.from_generator(lambda: (yield from train_dataset), features=train_dataset.features)
    else:
        train_dataset = load_dataset(f'ndc227/{args.dataset}', num_proc=num_proc, cache_dir='/nlp/scr/ayc227/.cache/huggingface/datasets')['train']

    if args.add_ids:
        model.add_ids = True
    if args.length_penalty > 0:
        model.length_penalty = args.length_penalty

    # train_dataset = train_dataset.map(sample_docs, num_proc=torch.cuda.device_count()) # Shouldn't need to downsample bc done in previous step
    # print_debug('check down_sample', len(train_dataset[0]['new_chunks']))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if not args.tiny:
        wandb.login(key='fc6c9280f011612e6aeb6c45fd6f79f7d08c56dc')
        wandb.init(
            # set the wandb project where this run will be logged
            entity='ndc227-stanford-university',
            project='new_chunking_experiments',
            # group='fsdp_run',

            # track hyperparameters and run metadata
            config={
            'learning_rate': args.lr,
            'architecture': 'Transformer',
            'dataset': 'NQ',
            'epochs': args.train_epochs,
            }
        )
        model.log_wandb = True

    if args.num_gpus > 0:
        trainer = L.Trainer(max_epochs=args.train_epochs, strategy='ddp_find_unused_parameters_true', accelerator='gpu', devices=args.num_gpus, default_root_dir='/nlp/scr/ayc227/lightning_logs')
    else:
        trainer = L.Trainer(max_epochs=args.train_epochs, default_root_dir='/nlp/scr/ayc227/lightning_logs')
    trainer.fit(model, train_loader)
    if args.save_name != None:
        model.push_to_hub(f'ndc227/{args.save_name}', private=True)

if args.eval:
    if args.eval_experiment == '4':
        model = ReplugTransformer.from_pretrained(f'ndc227/{args.eval_reranker}')

    if args.eval_experiment == '1' or args.eval_experiment == '1.5':
        eval_dataset = load_dataset(f'ndc227/{args.dataset}', split='test', num_proc=num_proc, cache_dir='/nlp/scr/ayc227/.cache/huggingface/datasets')
    elif args.eval_experiment == '2':
        eval_dataset = load_dataset(f'ndc227/{args.dataset}', args.exp2_subset, split='test', num_proc=num_proc, cache_dir='/nlp/scr/ayc227/.cache/huggingface/datasets')
    elif args.eval_experiment == '3':
        subsets = []
        configs = get_dataset_config_names(f'ndc227/{args.dataset}')
        for config in configs:
            eval_subset = load_dataset(f'ndc227/{args.dataset}', config, split='test', num_proc=num_proc, cache_dir='/nlp/scr/ayc227/.cache/huggingface/datasets')
            subsets.append(eval_subset)
        eval_dataset = subsets[0]
        eval_dataset = eval_dataset.map(combine_subsets, num_proc=num_proc, with_indices=True)
        # print_debug(eval_dataset, len(eval_dataset[0]['new_chunks']))
    elif args.eval_experiment == '3.5':
        eval_dataset = load_dataset(f'ndc227/{args.dataset}', split='test', num_proc=num_proc, cache_dir='/nlp/scr/ayc227/.cache/huggingface/datasets')
    elif args.eval_experiment == '4':
        eval_dataset = load_dataset(f'ndc227/{args.dataset}', split='test', num_proc=num_proc, cache_dir='/nlp/scr/ayc227/.cache/huggingface/datasets')
        eval_dataset = eval_dataset.remove_columns('retrieved')
        eval_dataset = eval_dataset.map(downsample_eval, num_proc=torch.cuda.device_count())
    else:
        quit(0)

    if args.tiny:
        eval_dataset = Dataset.from_dict(eval_dataset[:20])
        
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)

    model.evaluate(eval_loader, top_k=args.eval_k, experiment=args.eval_experiment)