import argparse
import os
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict, get_dataset_config_names, concatenate_datasets

# LOCAL USE:
# python combine_datasets.py --dataset rechunked_and_scored_nq --output_name toy_new_rechunked_and_scored_nq --split train --option 1
# python combine_datasets.py --dataset rechunked_and_scored_dev_nq --output_name combined_rechunked_and_scored_dev_nq --split train --option 1
# python combine_datasets.py --dataset dev_rechunked --output_name combined_dev_rechunked --split dev --option 2

argp = argparse.ArgumentParser()
argp.add_argument('--dataset', default='test_rechunked')
argp.add_argument('--user', default='ndc227')
argp.add_argument('--output_name', default='toy_new_rechunked_nq')
argp.add_argument('--split', default='train')
argp.add_argument('--save', action='store_true')
argp.add_argument('--option', default=1, type=int)
argp.add_argument('--debug', action='store_true')
args = argp.parse_args()

def combine_subset_chunks(ex, idx):
    for i in range(1, len(subsets)):
        ex['new_chunks'].extend(subsets[i][idx]['new_chunks'])
        ex['chunker_ids'].extend(subsets[i][idx]['chunker_ids'])
    return ex

def concatenate_subsets(ex, idx):
    return ex

os.environ['HF_TOKEN'] = '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'
num_proc = 1

subsets = []
configs = get_dataset_config_names(f'ndc227/{args.dataset}')
print(configs)
for config in configs:
    subset = load_dataset(f'ndc227/{args.dataset}', config, split=args.split, num_proc=num_proc, cache_dir='/nlp/scr/ayc227/.cache/huggingface/datasets')
    subsets.append(subset)
if args.option == 1:
    dataset = concatenate_datasets(subsets)
    dataset = DatasetDict({args.split: dataset})
elif args.option == 2:
    dataset = subsets[0]
    dataset = dataset.map(combine_subset_chunks, num_proc=num_proc, with_indices=True)
    dataset = DatasetDict({args.split: dataset})

print(dataset)
if args.save:
    dataset.push_to_hub(args.output_name, private='True')