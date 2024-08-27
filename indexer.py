import argparse
import math
import os
import sys

import numpy as np
import torch
from datasets import load_from_disk, load_dataset
from tqdm import tqdm, trange
from transformers import AutoModel, AutoTokenizer

from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_text_splitters import TokenTextSplitter, CharacterTextSplitter, NLTKTextSplitter, SentenceTransformersTokenTextSplitter, SpacyTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from chunking_evaluation.chunking import ClusterSemanticChunker, KamradtModifiedChunker
from chromadb.utils import embedding_functions

# python indexer.py --shard-id 0 --shards 8 --input new_chunks_with_retrieve --split train --output train_rechunked
# python indexer.py --shard-id 0 --shards 8 --input new_chunks_with_retrieve --split dev --output dev_rechunked

os.environ['HF_HOME'] = '/nlp/scr/ayc227/.cache/'
os.environ['HF_TOKEN'] = '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'

def parse_args(args):
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--batch-size",
    #     type=int,
    #     default=1024,
    #     required=True,
    #     help="Batch size",
    # )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        required=True,
        help="Zero-indexed shard number",
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=1,
        required=True,
        help="Number of shards",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        required=True,
        help="Path to dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        required=True,
        help="Split of input dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        required=True,
        help="Output path for shard.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args(args)
    return args


# def last_token_pool(last_hidden_states, attention_mask):
#     left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
#     if left_padding:
#         return last_hidden_states[:, -1]
#     else:
#         sequence_lengths = attention_mask.sum(dim=1) - 1
#         batch_size = last_hidden_states.shape[0]
#         return last_hidden_states[
#             torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
#         ]


def main(args):
    args = parse_args(args)
    np.random.seed(args.seed)

    total_shards = args.shards
    shard_id = args.shard_id
    if total_shards < 0:
        raise ValueError("Shards must be a positive integer.")
    if shard_id < 0 or shard_id >= total_shards:
        raise ValueError("Shard number must be between 0 and shards - 1.")

    # Get dataset
    # dataset = load_from_disk(os.path.expanduser(args.input))
    dataset = load_dataset(f'ndc227/{args.input}', split=args.split)

    chunks = dataset.num_rows
    shard_size = math.ceil(chunks / total_shards)
    start = shard_id * shard_size
    end = min((shard_id + 1) * shard_size, chunks)

    print(
        f"Processing shard {shard_id + 1} of {total_shards}, covering examples {start} to {end} out of {chunks}"
    )
    shard_dataset = dataset.select(range(start, end))

    # print(shard_dataset)
    # quit(0)

    # Process dataset
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
        chunk_size=500, chunk_overlap=0, add_start_index=True, separators=['\n\n', '\n', '.', '?', '!']
    )
    rec_char_splitter_4 = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, add_start_index=True
    )
    semantic_chunker = SemanticChunker(
        embeddings=HuggingFaceEmbeddings(), buffer_size=1, add_start_index=True
    )

    default_ef = embedding_functions.DefaultEmbeddingFunction()
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')
    kamradt_chunker = KamradtModifiedChunker(embedding_function=sentence_transformer_ef, avg_chunk_size=500)
    cluster_semantic_chunker_1 = ClusterSemanticChunker(embedding_function=sentence_transformer_ef, max_chunk_size=500)
    cluster_semantic_chunker_2 = ClusterSemanticChunker(embedding_function=sentence_transformer_ef, max_chunk_size=200)

    splitters = [rec_char_splitter_sentence, rec_char_splitter_2, semantic_chunker, cluster_semantic_chunker_1]
    if args.split != None:
        splitters = [rec_char_splitter_sentence]
    print('loaded splitters')

    def get_splits(splitter, chunks):
        splits = []
        for chunk in chunks:
            splits.extend(splitter.split_text(chunk))
        return splits

    def rechunk(ex):
        chunks = ex['retrieved']
        new_chunks = []
        chunkers = []
        for splitter in splitters:
            splits = get_splits(splitter, chunks)
            new_chunks.extend(splits)
            if f'{splitter.__class__.__name__}' in ['KamradtModifiedChunker']:
                chunkers.extend([f'{splitter.__class__.__name__}_{splitter.avg_chunk_size}_0' for _ in range(len(splits))])
            elif f'{splitter.__class__.__name__}' in ['ClusterSemanticChunker']:
                chunkers.extend([f'{splitter.__class__.__name__}_{splitter._chunk_size}_0' for _ in range(len(splits))])
            elif f'{splitter.__class__.__name__}' in ['SemanticChunker']:
                chunkers.extend([f'{splitter.__class__.__name__}_0_0' for _ in range(len(splits))])
            else:
                chunkers.extend([f'{splitter.__class__.__name__}_{splitter._chunk_size}_{splitter._chunk_overlap}' for _ in range(len(splits))])
        ex['new_chunks'] = new_chunks
        ex['chunker_ids'] = chunkers

        return ex
    
    rechunked = shard_dataset.map(rechunk)
    print(rechunked)
    # quit(0)

    # Save dataset
    splitter = splitters[0]
    if f'{splitter.__class__.__name__}' in ['KamradtModifiedChunker']:
        rechunked.push_to_hub(f'ndc227/{args.output}', f'{shard_id}_{splitter.__class__.__name__}_{splitter.avg_chunk_size}_0', private=True)
    elif f'{splitter.__class__.__name__}' in ['ClusterSemanticChunker']:
        rechunked.push_to_hub(f'ndc227/{args.output}', f'{shard_id}_{splitter.__class__.__name__}_{splitter._chunk_size}_0', private=True)
    else:
        rechunked.push_to_hub(f'ndc227/{args.output}', f'{shard_id}_{splitter.__class__.__name__}_{splitter._chunk_size}_{splitter._chunk_overlap}', private=True)

if __name__ == "__main__":
    main(sys.argv[1:])