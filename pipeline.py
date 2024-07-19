import argparse
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer
import wandb
import torch.cuda
import lightning as L

from replug_transformer import ReplugTransformer

argp = argparse.ArgumentParser()
argp.add_argument('--llm_name', default='facebook/opt-125m')
argp.add_argument('--train', default=False)
argp.add_argument('--eval', default=False)
argp.add_argument('--tiny', default=False)
argp.add_argument('--batch_size', default=8)
argp.add_argument('--train_epochs', default=1)
argp.add_argument('--lr', default=1e-4)
argp.add_argument('--train_set', default='train_chunks_with_retrieve')
argp.add_argument('--valid_set', default='valid_chunks_with_retrieve')
args = argp.parse_args()

os.environ['HF_TOKEN'] = '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'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset = load_dataset('json', data_files=f'data/{args.train_set}.json')['train']
valid_dataset = load_dataset('json', data_files=f'data/{args.valid_set}.json')['train']

if args.tiny:
    train_dataset = Dataset.from_dict(train_dataset[:20])
    valid_dataset = Dataset.from_dict(valid_dataset[:20])

batch_size = args.batch_size
model_id = args.llm_name
tokenizer = AutoTokenizer.from_pretrained(model_id)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

model = ReplugTransformer(model_id)

if args.train:
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

    trainer = L.Trainer(max_epochs=args.train_epochs)
    trainer.fit(model, train_loader, valid_loader)
    model.push_to_hub('ndc227/reranker_basic')

if args.eval:
    pass