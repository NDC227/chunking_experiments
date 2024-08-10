import argparse
import os
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter, CharacterTextSplitter, NLTKTextSplitter,\
                                     SentenceTransformersTokenTextSplitter, SpacyTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
from time import process_time
import json

from fast_bm25 import BM25

# Sample use:
# python rechunker.py --dataset new_chunks_with_retrieve --output_name rechunked_nq --num_proc 1
argp = argparse.ArgumentParser()
argp.add_argument('--dataset', default='new_chunks_with_retrieve')
argp.add_argument('--output_name', default='rechunked_nq')
argp.add_argument('--num_proc', default=1, type=int)
args = argp.parse_args()

os.environ['HF_HOME'] = '/nlp/scr/ayc227/.cache/'
os.environ['HF_TOKEN'] = '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'

num_proc = args.num_proc

# train_chunks = load_dataset(f'ndc227/{args.dataset}', split='train', streaming=True).take(100)
# train_chunks = Dataset.from_generator(lambda: (yield from train_chunks), features=train_chunks.features)
train_chunks = load_dataset(f'ndc227/{args.dataset}', split='train', num_proc=num_proc, cache_dir='/nlp/scr/ayc227/.cache/huggingface/datasets')
dev_chunks = load_dataset(f'ndc227/{args.dataset}', split='dev', num_proc=num_proc, cache_dir='/nlp/scr/ayc227/.cache/huggingface/datasets')
test_chunks = load_dataset(f'ndc227/{args.dataset}', split='test', num_proc=num_proc, cache_dir='/nlp/scr/ayc227/.cache/huggingface/datasets')
print('loaded chunks')

# print(train_chunks)

rec_char_splitter_1 = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=10, add_start_index=True
)
rec_char_splitter_2 = RecursiveCharacterTextSplitter(
    chunk_size=250, chunk_overlap=25, add_start_index=True
)
rec_char_splitter_3 = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50, add_start_index=True
)
rec_char_splitter_4 = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100, add_start_index=True
)
splitters = [rec_char_splitter_1, rec_char_splitter_2, rec_char_splitter_3, rec_char_splitter_4]
print('loaded splitters')

def get_splits(splitter, chunks):
    splits = []
    for chunk in chunks:
        splits.extend(splitter.split_text(chunk))
    return splits

def rechunk(ex):
    chunks = ex['retrieved']
    # print(chunks)
    new_chunks = []
    chunkers = []
    for splitter in splitters:
        splits = get_splits(splitter, chunks)
        new_chunks.extend(splits)
        chunkers.extend([f'{splitter.__class__.__name__}_{splitter._chunk_size}_{splitter._chunk_overlap}' for _ in range(len(splits))])
    ex['new_chunks'] = new_chunks
    ex['chunker_ids'] = chunkers

    return ex

train_chunks = train_chunks.map(rechunk, num_proc=num_proc)
print('train_chunks rechunked')
dev_chunks = dev_chunks.map(rechunk, num_proc=num_proc)
print('dev_chunks rechunked')
test_chunks = test_chunks.map(rechunk, num_proc=num_proc)
print('test_chunks rechunked')
print(len(train_chunks[0]['new_chunks']), len(train_chunks[0]['retrieved']))

combined_dataset = DatasetDict({'train':train_chunks, 'dev':dev_chunks, 'test':test_chunks})
combined_dataset.push_to_hub(f'ndc227/{args.output_name}', private=True)
print('done uploading to hub')