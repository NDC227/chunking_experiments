import argparse
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
from datasets import Dataset, get_dataset_config_names
import wandb
import torch.cuda
import torch.nn
import lightning as L
import numpy as np
import json

from replug_transformer_new import ReplugTransformer

# Reference BGE-M3 training hyperparameters from Chen et al. 2024

# SAMPLE USE:
# TRAINING:
# python new_pipeline.py --user {USER} --llm-name meta-llama/Meta-Llama-3.1-8B-Instruct --train-set scored_train_rechunked --dev-set scored_dev_rechunked --train --batch-size 2 --gradient-accumulation 8 --train-epochs 10 --lr 5e-5 --temperature 0.025 --save-name fine_tuned_reranker
# EVAL:
# python new_pipeline.py --user {USER} --llm-name meta-llama/Meta-Llama-3.1-8B-Instruct --dev-set chunks_with_retrieve --eval --eval-split dev --eval-experiment 1 --eval-outfile exp_1_results.json
# python new_pipeline.py --user {USER} --llm-name meta-llama/Meta-Llama-3.1-8B-Instruct --dev-set dev_rechunked --eval --eval-split dev --eval-experiment 2 --exp2-subset RecursiveCharacterTextSplitter_500_0 --eval-outfile exp_2_results.json
# python new_pipeline.py --user {USER} --llm-name meta-llama/Meta-Llama-3.1-8B-Instruct --dev-set scored_dev_rechunked --eval --eval-split dev --eval-experiment 3 --eval-outfile exp_3_results.json
# python new_pipeline.py --user {USER} --llm-name meta-llama/Meta-Llama-3.1-8B-Instruct --dev-set scored_dev_rechunked --eval --eval-split dev --eval-experiment 4 --eval-reranker fine_tuned_rechunker --eval-outfile exp_4_results.json

argp = argparse.ArgumentParser()
argp.add_argument(
    "--user", default="hf_user", help="Hugging Face username"
    )
argp.add_argument(
    "--llm-name", default="facebook/opt-125m", help="LLM to use for RAG generation"
    )
argp.add_argument(
    "--train-set", default="toy_rechunked_and_scored_nq", help="Train dataset"
    )
argp.add_argument(
    "--dev-set", default="valid_chunks_with_retrieve", help="Validation/Evaluation dataset"
    )
argp.add_argument(
    "--top-k", default=10, type=int, help="Top-k documents to use for generation in validation/eval"
    )
argp.add_argument(
    "--tiny", action="store_true", help="Run process on tiny dataset for debugging"
    )
argp.add_argument(
    "--num-proc", default=1, type=int, help="The number of processes to use to load datasets"
    )
argp.add_argument(
    "--cache-dir", default="/scratch", help="Cache directory"
    )
argp.add_argument(
    "--debug", action="store_true", help="Flag that, if used, gives extra debug prints"
    )

argp.add_argument(
    "--train", action="store_true", help="Flag for training"
    )
argp.add_argument(
    "--batch-size", default=2, type=int, help="Batch size for training/eval"
    )
argp.add_argument(
    "--gradient-accumulation", default=1, type=int, help="Number of gradient accumulation steps"
    )
argp.add_argument(
    "--train-epochs", default=1, type=int, help="Number of training epochs"
    )
argp.add_argument(
    "--lr", default=1e-5, type=float, help="Training learning rate"
    )
argp.add_argument(
    "--temperature", default=1.0, type=float, help="Softmax temperature"
    )
argp.add_argument(
    "--save-name", default=None, help="Name of fine-tuned reranker"
    )

argp.add_argument(
    "--eval", action="store_true", help="Flag for evaluation"
    )
argp.add_argument(
    "--eval-split", default="dev", help="Which split to run evaluation on (dev/test)"
    )
argp.add_argument(
    "--eval-experiment", default="1", help="Which experiment to run (1/1.5/2/3/3.5/4)"
    )
argp.add_argument(
    "--exp2-subset", default="", help="Which chunking strategy subset to use in experiment 2"
    )
argp.add_argument(
    "--eval-reranker", default=None, help="Which reranker to evaluate in experiment 4"
    )
argp.add_argument(
    "--eval-outfile", default="results.json", help="JSON file to output results"
    )

# for future experiments
argp.add_argument(
    "--add-ids", action="store_true", help="Add chunking strategy id (name, chunk size, overlap)"
    ) 
argp.add_argument(
    "--length-penalty", default=0.0, type=float, help="Add length penalty to the scores/loss"
    )

# defunct
argp.add_argument(
    "--num-gpus", default=0, type=int, help="Number of GPUs to use in training"
    ) 

args = argp.parse_args()

os.environ["WANDB_INIT_TIMEOUT"] = "300"
torch.set_float32_matmul_precision("medium")
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

llm_name = args.llm_name
num_proc = args.num_proc
batch_size = args.batch_size
gradient_accumulation = args.gradient_accumulation
cache_dir = args.cache_dir

def print_debug(*prints):
    if args.debug:
        print(prints)

def sample_docs(ex):
    idxs = torch.randint(len(ex["new_chunks"]), (5,))
    ex["new_chunks"] = np.take(ex["new_chunks"],idxs)
    ex["chunker_ids"] = np.take(ex["chunker_ids"],idxs)
    ex["llm_scores"] = np.take(ex["llm_scores"],idxs)
    return ex

def downsample_eval(ex):
    idxs = torch.randint(len(ex["new_chunks"]), (100,))
    ex["new_chunks"] = np.take(ex["new_chunks"],idxs)
    ex["chunker_ids"] = np.take(ex["chunker_ids"],idxs)
    return ex

model = ReplugTransformer(llm_name)
model.temperature = args.temperature
model.lr = args.lr
model.top_k = args.top_k
# model.llm_name = llm_name

if args.train:
    if args.tiny:
        train_dataset = load_dataset(f"{args.user}/{args.train_set}", split="train", streaming=True, cache_dir=f"{cache_dir}/huggingface/datasets").take(2)
        train_dataset = Dataset.from_generator(lambda: (yield from train_dataset), features=train_dataset.features)
        dev_dataset = load_dataset(f"{args.user}/{args.dev_set}", split="dev", streaming=True, cache_dir=f"{cache_dir}/huggingface/datasets").take(2)
        dev_dataset = Dataset.from_generator(lambda: (yield from dev_dataset), features=dev_dataset.features)
    else:
        train_dataset = load_dataset(f"{args.user}/{args.train_set}", num_proc=num_proc, cache_dir=f"{cache_dir}/huggingface/datasets")["train"]
        dev_dataset = load_dataset(f"{args.user}/{args.dev_set}", num_proc=num_proc, cache_dir=f"{cache_dir}/huggingface/datasets")["dev"]

    if args.add_ids:
        model.add_ids = True
    if args.length_penalty > 0:
        model.length_penalty = args.length_penalty

    if args.tiny:
        train_dataset = train_dataset.map(sample_docs, num_proc=torch.cuda.device_count()) # Shouldn"t need to downsample bc done in previous step
        dev_dataset = dev_dataset.map(sample_docs, num_proc=torch.cuda.device_count())
        print_debug("check down_sample", len(train_dataset[0]["new_chunks"]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    if args.tiny:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)

    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    if args.tiny:
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, drop_last=True)

    # if not args.tiny:
    if True:
        wandb.init(
            # set the wandb project where this run will be logged
            project="new_chunking_experiments",
            group="ddp_run",

            # track hyperparameters and run metadata
            config={
            "learning_rate": args.lr,
            "architecture": "Transformer",
            "dataset": "NQ",
            "epochs": args.train_epochs,
            }
        )
        model.log_wandb = True

    if torch.cuda.device_count() > 1:
        trainer = L.Trainer(max_epochs=args.train_epochs, strategy="deepspeed_stage_2", accelerator="gpu", devices=torch.cuda.device_count(), default_root_dir=f"{cache_dir}/lightning_logs", enable_checkpointing=False)
    else:
        trainer = L.Trainer(max_epochs=args.train_epochs, default_root_dir=f"{cache_dir}/lightning_logs", enable_checkpointing=False)
    trainer.accumulate_grad_batches = gradient_accumulation
    trainer.num_sanity_val_steps = 0
    trainer.fit(model, train_loader, dev_loader)
    if args.save_name != None:
        model.push_to_hub(f"{args.user}/{args.save_name}", private=True)

if args.eval:
    if args.eval_experiment == "1" or args.eval_experiment == "1.5":
        eval_dataset = load_dataset(f"{args.user}/{args.dev_set}", split=args.eval_split, num_proc=num_proc, cache_dir=f"{cache_dir}/huggingface/datasets")
    
    elif args.eval_experiment == "2":
        eval_dataset = load_dataset(f"{args.user}/{args.dev_set}", args.exp2_subset, split=args.eval_split, num_proc=num_proc, cache_dir=f"{cache_dir}/huggingface/datasets")
    
    elif args.eval_experiment == "3":
        eval_dataset = load_dataset(f"{args.user}/{args.dev_set}", split=args.eval_split, num_proc=num_proc, cache_dir=f"{cache_dir}/huggingface/datasets")
        eval_dataset = eval_dataset.map(downsample_eval, num_proc=torch.cuda.device_count())
    
    elif args.eval_experiment == "3.5":
        eval_dataset = load_dataset(f"{args.user}/{args.dev_set}", split=args.eval_split, num_proc=num_proc, cache_dir=f"{cache_dir}/huggingface/datasets")
    
    elif args.eval_experiment == "4":
        model = ReplugTransformer.from_pretrained(f"{args.user}/{args.eval_reranker}")
        model.llm_name = llm_name
        eval_dataset = load_dataset(f"{args.user}/{args.dev_set}", split=args.eval_split, num_proc=num_proc, cache_dir=f"{cache_dir}/huggingface/datasets")
        # eval_dataset = eval_dataset.remove_columns("retrieved")
        eval_dataset = eval_dataset.map(downsample_eval, num_proc=torch.cuda.device_count())
    
    else:
        quit(0)

    if args.tiny:
        eval_dataset = Dataset.from_dict(eval_dataset[:10])
        
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, drop_last=True)
    # print(next(iter(eval_dataset))["new_chunks"][:10])
    # quit(0)

    # Trying to parallelize evaluation...
    # class MyDataParallel(torch.nn.DataParallel):
    #     def __getattr__(self, name):
    #         try:
    #             return super().__getattr__(name)
    #         except AttributeError:
    #             return getattr(self.module, name)
            
    # if torch.cuda.device_count() > 1:
    #     model = MyDataParallel(model)

    eval_results = model.evaluate(llm_name, eval_loader, top_k=args.top_k, experiment=args.eval_experiment)
    with open(args.eval_outfile, "w") as file:
        json.dump(eval_results, file)

    # print(model.llm)
    # print(model.llm_tokenizer)
    # print(model.reranker)
    # print(model.reranker_tokenizer)
