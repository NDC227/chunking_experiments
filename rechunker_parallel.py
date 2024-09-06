import argparse
import math
# import os
import sys

import numpy as np
# import torch
from datasets import load_from_disk, load_dataset, Dataset
# from tqdm import tqdm, trange
# from transformers import AutoModel, AutoTokenizer

from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_text_splitters import TokenTextSplitter, CharacterTextSplitter, NLTKTextSplitter, SentenceTransformersTokenTextSplitter, SpacyTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from chunking_evaluation.chunking import ClusterSemanticChunker, KamradtModifiedChunker
from chromadb.utils import embedding_functions

# SAMPLE USE:
# python rechunker_parallel.py --shard-id 0 --shards 8 --user {USER} --input new_chunks_with_retrieve --split train --output split_train_rechunked
# python rechunker_parallel.py --shard-id 0 --shards 8 --user {USER} --input new_chunks_with_retrieve --split dev --output split_dev_rechunked

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard-id", type=int, default=0, required=True, help="Zero-indexed shard number",
    )
    parser.add_argument(
        "--shards", type=int, default=1, required=True, help="Number of shards",
    )
    parser.add_argument(
        "--user", default="hf_user", help="Hugging Face username"
    )
    parser.add_argument(
        "--input", type=str, default=None, required=True, help="Path to dataset",
    )
    parser.add_argument(
        "--split", type=str, default=None, required=True, help="Split of input dataset",
    )
    parser.add_argument(
        "--output", type=str, default=None, required=True, help="Output path for shard.",
    )
    parser.add_argument(
        "--cache-dir", default="/scratch", help="Cache directory"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed",
    )
    
    args = parser.parse_args(args)
    return args

def main(args):
    args = parse_args(args)
    np.random.seed(args.seed)
    cache_dir = args.cache_dir
    user = args.user

    total_shards = args.shards
    shard_id = args.shard_id
    if total_shards < 0:
        raise ValueError("Shards must be a positive integer.")
    if shard_id < 0 or shard_id >= total_shards:
        raise ValueError("Shard number must be between 0 and shards - 1.")

    # Get dataset
    train_chunks = load_dataset(f"{args.user}/{args.input}", split=args.split, cache_dir=cache_dir, streaming=True).skip(0).take(30000)
    dataset = Dataset.from_generator(lambda: (yield from train_chunks), features=train_chunks.features)
    dataset = load_dataset(f"{args.user}/{args.input}", cache_dir=cache_dir, split=args.split)

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
        chunk_size=500, chunk_overlap=0, add_start_index=True, separators=["\n\n", "\n", ".", "?", "!"]
    )
    rec_char_splitter_4 = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, add_start_index=True
    )
    semantic_chunker = SemanticChunker(
        embeddings=HuggingFaceEmbeddings(), buffer_size=1, add_start_index=True
    )

    default_ef = embedding_functions.DefaultEmbeddingFunction()
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    kamradt_chunker = KamradtModifiedChunker(embedding_function=sentence_transformer_ef, avg_chunk_size=500)
    kamradt_chunker_2 = KamradtModifiedChunker(embedding_function=sentence_transformer_ef, avg_chunk_size=250)
    kamradt_chunker_3 = KamradtModifiedChunker(embedding_function=sentence_transformer_ef, avg_chunk_size=50)
    cluster_semantic_chunker_1 = ClusterSemanticChunker(embedding_function=sentence_transformer_ef, max_chunk_size=500)
    cluster_semantic_chunker_2 = ClusterSemanticChunker(embedding_function=sentence_transformer_ef, max_chunk_size=250)
    cluster_semantic_chunker_3 = ClusterSemanticChunker(embedding_function=sentence_transformer_ef, min_chunk_size=10, max_chunk_size=50)

    splitters = [rec_char_splitter_sentence, rec_char_splitter_2, kamradt_chunker_3, cluster_semantic_chunker_3] # DEFUNCT
    # START: EDIT THIS FIELD FOR THE CHUNKING STRATEGY YOU WANT --------------------------------------------
    if args.split != None:
        splitters = [rec_char_splitter_sentence] 
    # END:   EDIT THIS FIELD FOR THE CHUNKING STRATEGY YOU WANT --------------------------------------------
    print("loaded splitters")

    def get_splits(splitter, chunks):
        splits = []
        for chunk in chunks:
            splits.extend(splitter.split_text(chunk))
        return splits

    def rechunk(ex):
        chunks = ex["retrieved"]
        new_chunks = []
        chunkers = []
        for splitter in splitters:
            splits = get_splits(splitter, chunks)
            new_chunks.extend(splits)
            if f"{splitter.__class__.__name__}" in ["KamradtModifiedChunker"]:
                chunkers.extend([f"{splitter.__class__.__name__}_{splitter.avg_chunk_size}_0" for _ in range(len(splits))])
            elif f"{splitter.__class__.__name__}" in ["ClusterSemanticChunker"]:
                chunkers.extend([f"{splitter.__class__.__name__}_{splitter._chunk_size}_0" for _ in range(len(splits))])
            elif f"{splitter.__class__.__name__}" in ["SemanticChunker"]:
                chunkers.extend([f"{splitter.__class__.__name__}_0_0" for _ in range(len(splits))])
            else:
                chunkers.extend([f"{splitter.__class__.__name__}_{splitter._chunk_size}_{splitter._chunk_overlap}" for _ in range(len(splits))])
        ex["new_chunks"] = new_chunks
        ex["chunker_ids"] = chunkers

        return ex
    
    rechunked = shard_dataset.map(rechunk)
    print(rechunked)
    # quit(0)

    # Save dataset
    splitter = splitters[0]
    if f"{splitter.__class__.__name__}" in ["KamradtModifiedChunker"]:
        rechunked.push_to_hub(f"{user}/{args.output}", f"{shard_id}_{splitter.__class__.__name__}_{splitter.avg_chunk_size}_0", private=True)
    elif f"{splitter.__class__.__name__}" in ["ClusterSemanticChunker"]:
        rechunked.push_to_hub(f"{user}/{args.output}", f"{shard_id}_{splitter.__class__.__name__}_{splitter._chunk_size}_0", private=True)
    else:
        rechunked.push_to_hub(f"{user}/{args.output}", f"{shard_id}_{splitter.__class__.__name__}_{splitter._chunk_size}_{splitter._chunk_overlap}", private=True)

if __name__ == "__main__":
    main(sys.argv[1:])