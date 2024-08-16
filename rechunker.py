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
# python rechunker.py --dataset new_chunks_with_retrieve --output_name toy_new_rechunked_nq --num_proc 1 --tiny
# python rechunker.py --dataset new_chunks_with_retrieve --output_name toy_test_rechunked --num_proc 1 --test_only --tiny
argp = argparse.ArgumentParser()
argp.add_argument('--dataset', default='new_chunks_with_retrieve')
argp.add_argument('--output_name', default='rechunked_nq')
argp.add_argument('--num_proc', default=1, type=int)
argp.add_argument('--test_only', action='store_true')
argp.add_argument('--tiny', action='store_true')
args = argp.parse_args()

os.environ['HF_HOME'] = '/nlp/scr/ayc227/.cache/'
os.environ['HF_TOKEN'] = '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'

num_proc = args.num_proc

if args.tiny:
    if not args.test_only:
        train_chunks = load_dataset(f'ndc227/{args.dataset}', split='train', streaming=True).take(1000)
        train_chunks = Dataset.from_generator(lambda: (yield from train_chunks), features=train_chunks.features)
        dev_chunks = load_dataset(f'ndc227/{args.dataset}', split='dev', streaming=True).take(10)
        dev_chunks = Dataset.from_generator(lambda: (yield from dev_chunks), features=dev_chunks.features)
    test_chunks = load_dataset(f'ndc227/{args.dataset}', split='test', streaming=True).take(500)
    test_chunks = Dataset.from_generator(lambda: (yield from test_chunks), features=test_chunks.features)

else:
    if not args.test_only:
        train_chunks = load_dataset(f'ndc227/{args.dataset}', split='train', num_proc=num_proc, cache_dir='/nlp/scr/ayc227/.cache/huggingface/datasets')
        dev_chunks = load_dataset(f'ndc227/{args.dataset}', split='dev', num_proc=num_proc, cache_dir='/nlp/scr/ayc227/.cache/huggingface/datasets')
    test_chunks = load_dataset(f'ndc227/{args.dataset}', split='test', num_proc=num_proc, cache_dir='/nlp/scr/ayc227/.cache/huggingface/datasets')
print('loaded chunks')

# print(train_chunks)

rec_char_splitter_1 = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=0, add_start_index=True
)
rec_char_splitter_2 = RecursiveCharacterTextSplitter(
    chunk_size=250, chunk_overlap=0, add_start_index=True
)
rec_char_splitter_3 = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=0, add_start_index=True
)
rec_char_splitter_sentence = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=0, add_start_index=True, separators=["\n\n", "\n", ".", "?", "!"]
)
rec_char_splitter_4 = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0, add_start_index=True
)

from chunking_evaluation.chunking import ClusterSemanticChunker, KamradtModifiedChunker
from chromadb.utils import embedding_functions
default_ef = embedding_functions.DefaultEmbeddingFunction()
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
kamradt_chunker = KamradtModifiedChunker(embedding_function=sentence_transformer_ef, avg_chunk_size=500)
cluster_semantic_chunker_1 = ClusterSemanticChunker(embedding_function=sentence_transformer_ef, max_chunk_size=500)
cluster_semantic_chunker_2 = ClusterSemanticChunker(embedding_function=sentence_transformer_ef, max_chunk_size=200)

splitters = [rec_char_splitter_sentence, rec_char_splitter_2, kamradt_chunker, cluster_semantic_chunker_1]
# splitters = [rec_char_splitter_1, rec_char_splitter_2, rec_char_splitter_3, rec_char_splitter_4]
# splitters = [rec_char_splitter_2]
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
        if f'{splitter.__class__.__name__}' in ['KamradtModifiedChunker']:
            chunkers.extend([f'{splitter.__class__.__name__}_{splitter.avg_chunk_size}_0' for _ in range(len(splits))])
        elif f'{splitter.__class__.__name__}' in ['ClusterSemanticChunker']:
            chunkers.extend([f'{splitter.__class__.__name__}_{splitter._chunk_size}_0' for _ in range(len(splits))])
        else:
            chunkers.extend([f'{splitter.__class__.__name__}_{splitter._chunk_size}_{splitter._chunk_overlap}' for _ in range(len(splits))])
    ex['new_chunks'] = new_chunks
    ex['chunker_ids'] = chunkers

    return ex

if not args.test_only:
    train_chunks = train_chunks.map(rechunk, num_proc=num_proc)
    print('train_chunks rechunked')
    print(train_chunks[0]['new_chunks'][:10])
    print(len(train_chunks[0]['new_chunks']), len(train_chunks[0]['retrieved']))
    dev_chunks = dev_chunks.map(rechunk, num_proc=num_proc)
    print('dev_chunks rechunked')
test_chunks = test_chunks.map(rechunk, num_proc=num_proc)
print('test_chunks rechunked')

if not args.test_only:
    combined_dataset = DatasetDict({'train':train_chunks, 'dev':dev_chunks, 'test':test_chunks})
    combined_dataset.push_to_hub(f'ndc227/{args.output_name}', private=True)
else:
    splitter = splitters[0]
    if f'{splitter.__class__.__name__}' in ['KamradtModifiedChunker']:
        test_chunks.push_to_hub(f'ndc227/{args.output_name}', f'{splitter.__class__.__name__}_{splitter.avg_chunk_size}_0', private=True)
    elif f'{splitter.__class__.__name__}' in ['ClusterSemanticChunker']:
        test_chunks.push_to_hub(f'ndc227/{args.output_name}', f'{splitter.__class__.__name__}_{splitter._chunk_size}_0', private=True)
    else:
        test_chunks.push_to_hub(f'ndc227/{args.output_name}', f'{splitter.__class__.__name__}_{splitter._chunk_size}_{splitter._chunk_overlap}', private=True)
    
print('done uploading to hub')