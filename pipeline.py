import argparse
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
from datasets import Dataset
import wandb
import torch.cuda
import lightning as L

from replug_transformer import ReplugTransformer

# Sample use
# python pipeline.py --llm_name facebook/opt-125m --train --batch_size 8 --train_epochs 1 --train_set chunks_retrieve_100_train --valid_set chunks_retrieve_100_valid
# python pipeline.py --llm_name facebook/opt-125m --train --batch_size 2 --train_epochs 1 --train_set chunks_retrieve_100_train --valid_set chunks_retrieve_100_valid
# python pipeline.py --llm_name facebook/opt-125m --train --batch_size 2 --train_epochs 1 --dataset new_chunks_with_retrieve
# python pipeline.py --llm_name microsoft/Phi-3-mini-4k-instruct --train --batch_size 2 --train_epochs 1 --dataset toy_chunks_with_retrieve
# python pipeline.py --llm_name facebook/opt-125m --eval --batch_size 8 --valid_set chunks_retrieve_100_valid --eval_k 10
# python pipeline.py --llm_name facebook/opt-125m --eval --batch_size 8 --dataset new_chunks_with_retrieve --eval_k 10

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
argp.add_argument('--train_set', default='train_chunks_with_retrieve')
argp.add_argument('--valid_set', default='valid_chunks_with_retrieve')
argp.add_argument('--dataset', default='new_chunks_with_retrieve')
argp.add_argument('--num_proc', default=1, type=int)
args = argp.parse_args()

os.environ['HF_TOKEN'] = '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'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# train_dataset = load_dataset('json', data_files=f'data/{args.train_set}.json')['train']
# valid_dataset = load_dataset('json', data_files=f'data/{args.valid_set}.json')['train']

batch_size = args.batch_size
model_id = args.llm_name
num_proc = args.num_proc

model = ReplugTransformer(model_id)

if args.train:
    # train_dataset = load_dataset(f'ndc227/{args.train_set}', streaming=True)['train']
    # valid_dataset = load_dataset(f'ndc227/{args.valid_set}', streaming=True)['train']

    train_dataset = load_dataset(f'ndc227/{args.dataset}', num_proc=num_proc)['train']
    valid_dataset = load_dataset(f'ndc227/{args.dataset}', num_proc=num_proc)['dev']

    # train_dataset = train_dataset.rename_columns({'query':'questions', 'gold_generation':'answers'})

    if args.tiny:
        train_dataset = Dataset.from_dict(train_dataset[:20])
        valid_dataset = Dataset.from_dict(valid_dataset[:20])
        
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    wandb.login(key='fc6c9280f011612e6aeb6c45fd6f79f7d08c56dc')
    wandb.init(
        # set the wandb project where this run will be logged
        entity='ndc227-stanford-university',
        project='chunking_experiments',

        # track hyperparameters and run metadata
        config={
        'learning_rate': args.lr,
        'architecture': 'Transformer',
        'dataset': 'NQ',
        'epochs': args.train_epochs,
        }
    )

    # trainer = L.Trainer(max_epochs=args.train_epochs, accelerator='gpu', devices=4)
    trainer = L.Trainer(max_epochs=args.train_epochs)
    trainer.fit(model, train_loader, valid_loader)
    if args.save:
        model.push_to_hub('ndc227/reranker_basic3', private=True)

if args.eval:
    if not args.baseline:
        model = ReplugTransformer.from_pretrained("ndc227/reranker_basic2")
    # valid_dataset = load_dataset(f'ndc227/{args.valid_set}', streaming=True)['train']
    valid_dataset = load_dataset(f'ndc227/{args.dataset}', num_proc=num_proc)['test']
    valid_dataset = valid_dataset.shuffle()

    if args.tiny:
        valid_dataset = Dataset.from_dict(valid_dataset[:20])
        
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    if args.baseline:
        model.eval(valid_loader, top_k=args.eval_k, rerank=False)
    else:
        model.eval(valid_loader, top_k=args.eval_k)