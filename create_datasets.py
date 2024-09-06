import argparse
import os
from time import process_time
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets

# import numpy as np
# from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter, CharacterTextSplitter, NLTKTextSplitter,\
#                                      SentenceTransformersTokenTextSplitter, SpacyTextSplitter
# from langchain_experimental.text_splitter import SemanticChunker
# from langchain_huggingface import HuggingFaceEmbeddings
# from tqdm import tqdm
# import json

from fast_bm25 import BM25

# SAMPLE USE:
# python create_datasets.py --user {USER} --cache-dir /scratch --output-name chunks_with_retrieve --retrieve-k 100 --num-proc 1 --tiny

argp = argparse.ArgumentParser()
argp.add_argument(
    "--user", default="hf_user", help="Hugging Face username"
    )
argp.add_argument(
    "--output-name", default="chunks", help="Name of output dataset"
    )
argp.add_argument(
    "--cache-dir", default="/scratch", help="Cache directory"
    )
argp.add_argument(
    "--retrieve-k", default=10, type=int, help="Top-K value"
    )
argp.add_argument(
    "--num-proc", default=1, type=int, help="Number of multiprocessing units"
    )
argp.add_argument(
    "--tiny", action="store_true", help="Run on a tiny dataset for debugging"
    )

# DEFUNCT ARGUMENTS
argp.add_argument(
    "--chroma", action="store_true", help="Use Chroma instead of BM25 for retrieval"
    ) 
argp.add_argument(
    "--reload-chunks", action="store_true", help="Generate chunks again even if exists"
    )
argp.add_argument(
    "--iterable", action="store_true", help="Use IterableDataset for data streaming"
    )

args = argp.parse_args()

def get_splits(splitter, data):
    return [split.page_content for split in splitter.split_documents(data)]

def retrieve_from_query(query, k, return_idx=False):
    tokenized_query = query.split() 
    if return_idx:
        retrieve = bm25.get_top_n(tokenized_query, return_idx=return_idx, n=k)
        return retrieve
    retrieve = bm25.get_top_n(tokenized_query, tokenized_splits, return_idx=return_idx, n=k)
    retrieved_docs = [" ".join(x) for x in retrieve]
    return retrieved_docs

# For (fast) BM25
def add_retrieval_results(ex):
    ex["retrieved"] = retrieve_from_query(ex["query"], k=args.retrieve_k)
    ex["gold_generation"] = ex["gold_generation"][0]
    return ex

# For Chroma - Not recommended, very slow
def add_retrieval_results_chroma(ex):
    ex["retrieved"] = collection.query(query_texts=[ex["query"]], n_results=args.retrieve_k)
    ex["gold_generation"] = ex["gold_generation"][0]
    return ex

# Data pre-processing (tokenize and de-nesting)
def preprocess(ex):
    ex["questions"] = ex["questions"][0]["input_text"]
    ex["answers"] = ex["answers"][0]["span_text"]
    return ex

user = args.user
num_proc = args.num_proc
cache_dir = args.cache_dir

if args.tiny:
    iterable_ds = load_dataset("ContextualAI/wiki_dpr_mapped_nq_field", split="train", streaming=True).take(100)
    wiki = Dataset.from_generator(lambda: (yield from iterable_ds), features=iterable_ds.features)
else:
    wiki = load_dataset("ContextualAI/wiki_dpr_mapped_nq_field", split="train", num_proc=num_proc, cache_dir=f"{cache_dir}/datasets")
print("loaded wikipedia")

OPTION = 1
if OPTION == 1:
    # tokenized_splits = [doc.split() for ex in tqdm(wiki) for doc in ex["passages"]]
    if not args.chroma:
        tokenized_splits = wiki.map(lambda ex: {"passages":[doc.split() for doc in ex["passages"]]}, num_proc=num_proc)["passages"]
        tokenized_splits = [chunk for article in tokenized_splits for chunk in article]
    else:
        tokenized_splits = [chunk for article in wiki["passages"] for chunk in article]
    all_ids = [str(i) for i in range(len(tokenized_splits))]
    del wiki
    # quit(0)
    # print(tokenized_splits[:10])
    print("chunks loaded")

    # print(docs[:10])
    start = process_time()
    if args.chroma:
        import chromadb
        chroma_client = chromadb.Client()
        collection = chroma_client.get_or_create_collection(name="wiki_chunks")
        print("collection made")
        # print(tokenized_splits[:2])
        # print(all_ids[:2])
        collection.upsert(
            ids=all_ids,
            documents=tokenized_splits
        )
        print("chroma loaded")
    else:
        if not os.path.isfile(f"bm25_{args.output_name}_1.pkl"):
            bm25 = BM25(tokenized_splits)
            bm25.save(f"bm25_{args.output_name}_1")
            print("bm25 loaded")
        else:
            bm25 = BM25.load(f"bm25_{args.output_name}_1")
    stop = process_time()
    print("time elapsed:", stop - start)

    train_dataset = load_dataset("ContextualAI/nq", split="train", cache_dir=f"{cache_dir}/datasets", num_proc=num_proc)
    print("train dataset loaded")
    if args.chroma:
        train_dataset = train_dataset.map(add_retrieval_results_chroma, num_proc=num_proc)
    else:
        train_dataset = train_dataset.map(add_retrieval_results, num_proc=num_proc)
    train_dataset = train_dataset.rename_columns({"query":"questions", "gold_generation":"answers"})
    train_dataset = train_dataset.filter(lambda ex: len(ex["retrieved"]) == args.retrieve_k, num_proc=num_proc)
    print("train dataset processed")
    # quit(0)

    dev_dataset = load_dataset("ContextualAI/nq", split="dev", cache_dir=f"{cache_dir}/datasets", num_proc=num_proc)
    dev_dataset = dev_dataset.map(add_retrieval_results, num_proc=num_proc)
    dev_dataset = dev_dataset.rename_columns({"query":"questions", "gold_generation":"answers"})
    dev_dataset = dev_dataset.filter(lambda ex: len(ex["retrieved"]) == args.retrieve_k, num_proc=num_proc)

    test_dataset = load_dataset("ContextualAI/nq", split="test", cache_dir=f"{cache_dir}/datasets", num_proc=num_proc)
    test_dataset = test_dataset.map(add_retrieval_results, num_proc=num_proc)
    test_dataset = test_dataset.rename_columns({"query":"questions", "gold_generation":"answers"})
    test_dataset = test_dataset.filter(lambda ex: len(ex["retrieved"]) == args.retrieve_k, num_proc=num_proc)
    print("all datasets processed")

    combined_dataset = DatasetDict({"train":train_dataset, "dev":dev_dataset, "test":test_dataset})
    combined_dataset.push_to_hub(f"{user}/{args.output_name}", private=True)
    # quit(0)

# ALL BELOW DEFUNCT!!! ---------------------------------------------- SEE rechunker.py/rechunker_parallel.py
# elif OPTION == 2:
    # class DocWrapper(object):
    #   def __init__(self, s):
    #     self.page_content = s
    #     self.metadata = {}

    # For (fast) BM25 with chunkers
    # def add_retrieval_results_langchain(ex):
    #     idxs = retrieve_from_query(ex["query"], return_idx=True, k=args.retrieve_k)
    #     ex["retrieved"] = np.asarray(all_splits).take(idxs)
    #     ex["ids"] = np.asarray(all_ids).take(idxs)
    #     ex["gold_generation"] = ex["gold_generation"][0]
    #     return ex

    # rec_char_splitter_1 = RecursiveCharacterTextSplitter(
    #     chunk_size=1000, chunk_overlap=100, add_start_index=True
    # )
    # rec_char_splitter_2 = RecursiveCharacterTextSplitter(
    #     chunk_size=2000, chunk_overlap=200, add_start_index=True
    # )
    # rec_char_splitter_3 = RecursiveCharacterTextSplitter(
    #     chunk_size=4000, chunk_overlap=200, add_start_index=True
    # )
    # rec_char_splitter_4 = RecursiveCharacterTextSplitter(
    #     chunk_size=8000, chunk_overlap=200, add_start_index=True
    # )
    # rec_char_splitter_5 = RecursiveCharacterTextSplitter(
    #     chunk_size=1000, chunk_overlap=400, add_start_index=True
    # )
    # rec_char_splitter_6 = RecursiveCharacterTextSplitter(
    #     chunk_size=2000, chunk_overlap=400, add_start_index=True
    # )
    # rec_char_splitter_7 = RecursiveCharacterTextSplitter(
    #     chunk_size=4000, chunk_overlap=800, add_start_index=True
    # )
    # rec_char_splitter_8 = RecursiveCharacterTextSplitter(
    #     chunk_size=8000, chunk_overlap=800, add_start_index=True
    # )
    # # token_splitter = TokenTextSplitter(
    # #     encoding_name="gpt2", chunk_size=100, chunk_overlap=0
    # # )
    # # char_splitter = CharacterTextSplitter(
    # #     separator="\n", is_separator_regex=False
    # # )
    # # nltk_splitter = NLTKTextSplitter(
    # #     separator="\n\n", language="english"
    # # )
    # # sentence_transformer_splitter = SentenceTransformersTokenTextSplitter(
    # #     chunk_overlap=50, model_name="sentence-transformers/all-mpnet-base-v2", tokens_per_chunk=None
    # # )
    # # spacy_splitter = SpacyTextSplitter(
    # #     separator="\n\n", pipeline="en_core_web_sm", max_length=1000000
    # # )
    # # semantic_chunker = SemanticChunker(
    # #     embeddings=HuggingFaceEmbeddings(), buffer_size=1, add_start_index=True
    # # )
    
    # #... use different parameters for these (especially chunk sizes) # TODO: How wide of a parameter space should we use? Default is 4000/200

    # # splitters = [rec_char_splitter_1, rec_char_splitter_2, token_splitter, char_splitter, nltk_splitter, \
    #              # sentence_transformer_splitter, spacy_splitter, semantic_chunker]
    # # splitters = [rec_char_splitter_1, token_splitter]
    # splitters = [rec_char_splitter_1, rec_char_splitter_2, rec_char_splitter_3, rec_char_splitter_4, \
    #              rec_char_splitter_5, rec_char_splitter_6, rec_char_splitter_7, rec_char_splitter_8] 
    # if args.reload_chunks:
    #     # docs = [DocWrapper(" ".join(ex["passages"])) for ex in tqdm(wiki)]
    #     if args.iterable:
    #         docs = iterable_ds.map(lambda ex: {"passages": " ".join(ex["passages"])})#["passages"]
    #     else:
    #         docs = wiki.map(lambda ex: {"passages": " ".join(ex["passages"])}, num_proc=num_proc)#["passages"]
    #         # print(len(docs[:10]))
    #         # quit(0)
    #         del wiki
        
    #     all_splits, all_ids = [], []

    #     if args.iterable:
    #         def split_text(ex):
    #             for i in range(len(splitters)):
    #                 ex[f"chunks{i}"] = splitters[i].split_text(ex["passages"])
    #             return ex
    #         splits = docs.map(split_text, remove_columns=["passages", "uuid", "title", "original_start_id", "original_passages", "from_nq"])#["retrieved"]
    #         for split in splits:
    #             # print(split)
    #             for splitter in split:
    #                 # print(split[splitter])
    #                 all_splits.extend(split[splitter])
    #                 all_ids.extend(splitter.__class__.__name__ for i in range(len(split[splitter])))

    #         chunks = Dataset.from_dict({"chunk":all_splits, "id":all_ids})
    #         chunks.push_to_hub("ndc227/toy_langchain_chunks", private=True)
    #         del docs
    #     else:
    #         parquet_lengths = {}
    #         for splitter in splitters:
    #             if splitter.__class__.__name__ != "SemanticChunker":
    #                 print(f"{splitter.__class__.__name__}_{splitter._chunk_size}_{splitter._chunk_overlap}")
    #             else:
    #                 print(f"{splitter.__class__.__name__}")
    #             # splits = get_splits(splitter, docs)

    #             # [split.page_content for split in splitter.split_documents(data)]
    #             # all_splits.extend(splits)
    #             # all_ids.extend([f"id_{splitter.__class__.__name__}_{ii}" for ii in range(len(splits))])
    #             # all_ids.extend([splitter.__class__.__name__ for ii in range(len(splits))])

    #             splits = docs.map(lambda ex: {"chunks": splitter.split_text(ex["passages"])}, \
    #                               remove_columns=["passages", "uuid", "title", "original_start_id", "original_passages", "from_nq"], num_proc=num_proc)#["retrieved"]
    #             splits = [x for chunk in splits for x in chunk["chunks"]]
    #             ids = [splitter.__class__.__name__ for i in range(len(splits))]
    #             parquet_lengths[f"{splitter.__class__.__name__}_{splitter._chunk_size}_{splitter._chunk_overlap}"] = len(splits)
    #             chunks = Dataset.from_dict({"chunks":splits, "ids":ids})
    #             chunks.push_to_hub(f"ndc227/toy_{splitter.__class__.__name__}_{splitter._chunk_size}_{splitter._chunk_overlap}", private=True)
    #             chunks.to_parquet(f"/nlp/scr/ayc227/dataset_parquets/toy_{splitter.__class__.__name__}_{splitter._chunk_size}_{splitter._chunk_overlap}")

    #             del splits, ids, chunks
            
    #         with open("parquet_lengths.json", "w") as file:
    #             json.dump(parquet_lengths, file)
    #             # print(splits[:10])
    #             # for split in splits:
    #             #     all_splits.extend(split["retrieved"])
    #             #     all_ids.extend([splitter.__class__.__name__ for i in range(len(split["retrieved"]))])
    #             # print(all_splits[:10])
    #             # quit(0)
    #         del docs
        
    # if True:
    #     try:
    #         if args.reload_chunks or not os.path.isfile(f"bm25_{args.output_name}_2.pkl"):
    #             print("creating bm25")
    #             bm25 = BM25()
    #             for splitter in splitters:
    #                 print(f"{splitter.__class__.__name__}_{splitter._chunk_size}_{splitter._chunk_overlap}")
    #                 chunks = load_dataset(f"ndc227/toy_{splitter.__class__.__name__}_{splitter._chunk_size}_{splitter._chunk_overlap}", \
    #                                         split="train", cache_dir="/nlp/scr/ayc227/.cache/huggingface/datasets", num_proc=num_proc)
    #                 tokenized_chunks = [x.split() for chunk in chunks for x in chunk["chunks"]]
    #                 # print(tokenized_chunks[:2])
    #                 bm25.update_corpus(tokenized_chunks)
    #             bm25.filter_corpus()
    #             bm25.save(f"bm25_{args.output_name}_2")
    #         else:
    #             print("loading bm25 from disk")
    #             bm25 = BM25.load(f"bm25_{args.output_name}_2")
    #             print(len(bm25.doc_len))
    #     except:
    #         raise RuntimeError("Chunk dataset does not exist")
    # else:
    #     try:
    #         all_splits, all_ids = [], []
    #         for splitter in splitters:
    #             # print(f"ndc227/toy_{splitter.__class__.__name__}_{splitter._chunk_size}_{splitter._chunk_overlap}")
    #             chunks = load_dataset(f"ndc227/toy_{splitter.__class__.__name__}_{splitter._chunk_size}_{splitter._chunk_overlap}", \
    #                                     split="train", cache_dir="/nlp/scr/ayc227/.cache/huggingface/datasets", num_proc=num_proc)
    #             for chunk in chunks:
    #                 all_splits.extend(chunk["chunks"])
    #                 all_ids.extend(chunk["ids"])
    #     except:
    #         raise RuntimeError("Chunk dataset does not exist")
    #     # df = pd.DataFrame.from_dict({"chunk": all_splits, "id": all_ids})
    #     # df.to_json("data/chunks.json")

    #     print(all_splits[:10])
    #     tokenized_splits = [x.split() for x in all_splits]
    #     print("data tokenized")
    #     if args.reload_chunks or not os.path.isfile(f"bm25_{args.output_name}_2.pkl"):
    #         print("creating bm25")
    #         bm25 = BM25(tokenized_splits)
    #         bm25.save(f"bm25_{args.output_name}_2")
    #     else:
    #         bm25 = BM25.load(f"bm25_{args.output_name}_2")
    #     print("bm25 loaded")

    # # all_splits, all_ids = [], []
    # # for splitter in splitters:
    # #     chunks = load_dataset(f"ndc227/toy_{splitter.__class__.__name__}_{splitter._chunk_size}_{splitter._chunk_overlap}", \
    # #                             split="train", cache_dir="/nlp/scr/ayc227/.cache/huggingface/datasets", num_proc=num_proc)
    # #     print(chunks["chunks"])
    # #     all_splits.extend([chunk["chunks"]["retrieved"] for chunk in chunks])
    # #     all_ids.extend([f"{splitter.__class__.__name__}_{splitter._chunk_size}_{splitter._chunk_overlap}" for chunk in chunks for x in chunk])
    # # print(all_splits)

    # file = open("parquet_lengths.json")
    # parquet_lengths = json.load(file)
    # parquet_length_values = list(parquet_lengths.values())
    # parquet_id_starts = [0]
    # for i in range(1, len(parquet_length_values)):
    #     parquet_id_starts.append(parquet_id_starts[i-1] + parquet_length_values[i-1])
    # print(parquet_id_starts)

    # train_dataset = load_dataset("ContextualAI/nq", split="train", cache_dir="/nlp/scr/ayc227/.cache/huggingface/datasets", num_proc=num_proc)
    # train_dataset = train_dataset.map(add_retrieval_results_langchain, num_proc=num_proc)
    # train_dataset = train_dataset.rename_columns({"query":"questions", "gold_generation":"answers"})
    # train_dataset = train_dataset.filter(lambda ex: len(ex["retrieved"]) == args.retrieve_k, num_proc=num_proc)
    # print("train dataset processed")

    # dev_dataset = load_dataset("ContextualAI/nq", split="dev", cache_dir="/nlp/scr/ayc227/.cache/huggingface/datasets", num_proc=num_proc)
    # dev_dataset = dev_dataset.map(add_retrieval_results_langchain, num_proc=num_proc)
    # dev_dataset = dev_dataset.rename_columns({"query":"questions", "gold_generation":"answers"})
    # dev_dataset = dev_dataset.filter(lambda ex: len(ex["retrieved"]) == args.retrieve_k, num_proc=num_proc)

    # test_dataset = load_dataset("ContextualAI/nq", split="test", cache_dir="/nlp/scr/ayc227/.cache/huggingface/datasets", num_proc=num_proc)
    # test_dataset = test_dataset.map(add_retrieval_results_langchain, num_proc=num_proc)
    # test_dataset = test_dataset.rename_columns({"query":"questions", "gold_generation":"answers"})
    # test_dataset = test_dataset.filter(lambda ex: len(ex["retrieved"]) == args.retrieve_k, num_proc=num_proc)
    # print("all datasets processed")

    # combined_dataset = DatasetDict({"train":train_dataset, "dev":dev_dataset, "test":test_dataset})
    # combined_dataset.push_to_hub(f"ndc227/{args.output_name}", private=True)