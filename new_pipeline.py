import argparse
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
from datasets import Dataset
import wandb
import torch.cuda
import lightning as L
import numpy as np

from replug_transformer_new import ReplugTransformer

# Sample use
# python new_pipeline.py --train --tiny --dataset toy_rechunked_and_scored_nq --train_epochs 2 --batch_size 2
argp = argparse.ArgumentParser()
argp.add_argument('--llm_name', default='facebook/opt-125m')
argp.add_argument('--train', action='store_true')
argp.add_argument('--eval', action='store_true')
argp.add_argument('--tiny', action='store_true')
argp.add_argument('--save', action='store_true')
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
args = argp.parse_args()

os.environ["WANDB_INIT_TIMEOUT"] = "300"
os.environ['HF_TOKEN'] = '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'
torch.set_float32_matmul_precision('medium')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_id = args.llm_name
num_proc = args.num_proc
batch_size = args.batch_size

model = ReplugTransformer(model_id)

def sample_docs(ex):
    idxs = torch.randint(len(ex['new_chunks']), (100,))
    ex['new_chunks'] = np.take(ex['new_chunks'],idxs)
    ex['chunker_ids'] = np.take(ex['chunker_ids'],idxs)
    ex['llm_scores'] = np.take(ex['llm_scores'],idxs)
    return ex

if args.train:
    if args.tiny:
        train_dataset = load_dataset(f'ndc227/{args.dataset}', split='train', streaming=True, cache_dir='/nlp/scr/ayc227/.cache/huggingface/datasets').take(20)
        train_dataset = Dataset.from_generator(lambda: (yield from train_dataset), features=train_dataset.features)
    else:
        train_dataset = load_dataset(f'ndc227/{args.dataset}', num_proc=num_proc, cache_dir='/nlp/scr/ayc227/.cache/huggingface/datasets')['train']

    train_dataset = train_dataset.map(sample_docs, num_proc=torch.cuda.device_count())
    # print(train_dataset[0])
    print(len(train_dataset[0]['new_chunks']))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if args.num_gpus > 0:
        trainer = L.Trainer(max_epochs=args.train_epochs, accelerator='gpu', devices=args.num_gpus, default_root_dir="/nlp/scr/ayc227/lightning_logs")
    else:
        trainer = L.Trainer(max_epochs=args.train_epochs, default_root_dir="/nlp/scr/ayc227/lightning_logs")
    trainer.fit(model, train_loader)
    if args.save:
        model.push_to_hub('ndc227/reranker_basic3', private=True)