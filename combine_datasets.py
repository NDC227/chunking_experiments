import argparse
# import os
# import numpy as np
from datasets import load_dataset, Dataset, DatasetDict, get_dataset_config_names, concatenate_datasets

# SAMPLE USE:
# python combine_datasets.py --user {USER} --dataset toy_train_rechunked --config RecursiveCharacterTextSplitter_500_0 --split train --output-name train_rechunked --option 1
# python combine_datasets.py --user {USER} --dataset train_rechunked --split train --output-name combined_train_rechunked --option 2

argp = argparse.ArgumentParser()
argp.add_argument(
    "--user", default="hf_user", help="Hugging Face username"
    )
argp.add_argument(
    "--dataset", default="test_rechunked", help="Input dataset name"
    )
argp.add_argument(
    "--config", default=None, help="For option 1, choose which chunker to combine over"
    )
argp.add_argument(
    "--split", default="train", help="Which train/dev/test split to combine over"
    )
argp.add_argument(
    "--output-name", default="toy_new_rechunked_nq", help="Name of the output dataset"
    )
argp.add_argument(
    "--option", default=1, type=int, help="Choose 1 for combining shards of single chunker; 2 for combining multiple chunker results"
    )
argp.add_argument(
    "--save", action="store_true", help="Flag that if used, saves the combined dataset to HF Hub; if not, NOTHING IS SAVED"
    )
argp.add_argument(
    "--num-proc", default=1, type=int, help="Number of multiprocessing units"
    )
argp.add_argument(
    "--cache-dir", default="/scratch", help="Cache directory"
    )
# argp.add_argument(
#     "--debug", action="store_true", help="Flag that if used, adds some debug prints"
#     )
args = argp.parse_args()

def combine_subset_chunks(ex, idx):
    for i in range(1, len(subsets)):
        ex["new_chunks"].extend(subsets[i][idx]["new_chunks"])
        ex["chunker_ids"].extend(subsets[i][idx]["chunker_ids"])
    return ex

num_proc = args.num_proc
user = args.user
cache_dir = args.cache_dir

if args.option == 1:
    combined_datasets = []
    configs = get_dataset_config_names(f"{user}/{args.dataset}")
    subsets = []
    
    for curr_config in configs:
        if args.config in curr_config:
            subset = load_dataset(f"{user}/{args.dataset}", curr_config, split=args.split, num_proc=num_proc, cache_dir=f"{cache_dir}/datasets")
            subsets.append(subset)
    dataset = concatenate_datasets(subsets)
    print(dataset)
    
elif args.option == 2:
    # START: EDIT THESE VALUES -----------------------------------------------
    configs = ["RecursiveCharacterTextSplitter_500_0", "RecursiveCharacterTextSplitter_250_0", "ClusterSemanticChunker_50_0", "KamradtModifiedChunker_500_0"]
    # END:   EDIT THESE VALUES -----------------------------------------------
    combined_datasets = []
    subsets = []
    for curr_config in configs:        
        subset = load_dataset(f"{user}/{args.dataset}", curr_config, split=args.split, num_proc=num_proc, cache_dir=f"{cache_dir}/datasets")
        subsets.append(subset)

    dataset = subsets[0]
    dataset = dataset.map(combine_subset_chunks, num_proc=num_proc, with_indices=True)
    dataset = DatasetDict({args.split: dataset})

    print(dataset)
    
else:
    quit(1)

if args.save:
    if args.option == 1:
        dataset.push_to_hub(args.output_name, args.config, private="True")
    else:
        dataset.push_to_hub(args.output_name, private="True")
