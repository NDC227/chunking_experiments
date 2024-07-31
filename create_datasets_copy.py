import argparse
import os
from datasets import load_dataset, Dataset, DatasetDict
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter, CharacterTextSplitter, NLTKTextSplitter,\
                                     SentenceTransformersTokenTextSplitter, SpacyTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

from fast_bm25 import BM25

# Sample use:
# python create_datasets_copy.py --dir data --output_name chunks_retrieve_100 --retrieve_k 100
# python create_datasets_copy.py --dir data --output_name new_chunks_with_retrieve --retrieve_k 100
argp = argparse.ArgumentParser()
argp.add_argument('--dir', default='data')
argp.add_argument('--output_name', default='chunks')
argp.add_argument('--retrieve_k', default=10, type=int)
argp.add_argument('--num_proc', default=1, type=int)
args = argp.parse_args()

os.environ['HF_TOKEN'] = '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'

class DocWrapper(object):
  def __init__(self, s):
    self.page_content = s
    self.metadata = {}

def get_splits(splitter, data):
    return [split.page_content for split in splitter.split_documents(data)]

def retrieve_from_query(query, k):
    tokenized_query = query.split() 
    retrieve = bm25.get_top_n(tokenized_query, tokenized_splits, n=k)
    retrieved_docs = [' '.join(x) for x in retrieve]
    return retrieved_docs

# For (fast) BM25
def add_retrieval_results(ex):
    ex['retrieved'] = retrieve_from_query(ex['query'], k=args.retrieve_k)
    ex['gold_generation'] = ex['gold_generation'][0]
    return ex

# def add_retrieval_results_2(ex):
#     ex['retrieved'] = retrieve_from_query(ex['query'], k=args.retrieve_k)
#     ex['gold_generation'] = ex['gold_generation'][0]
#     return ex

# Data pre-processing (tokenize and de-nesting)
def preprocess(ex):
    ex['questions'] = ex['questions'][0]['input_text']
    ex['answers'] = ex['answers'][0]['span_text']
    return ex

num_proc = args.num_proc
retrieve_k = args.retrieve_k
# iterable_ds = load_dataset('ContextualAI/wiki_dpr_mapped_nq_field', split='train', streaming=True).take(10000)
# wiki = Dataset.from_generator(lambda: (yield from iterable_ds), features=iterable_ds.features)
wiki = load_dataset('ContextualAI/wiki_dpr_mapped_nq_field', split='train', num_proc=num_proc).take(1000)
print('loaded wikipedia')

OPTION = 1
if OPTION == 1:
    # tokenized_splits = [doc.split() for ex in tqdm(wiki) for doc in ex['passages']]
    tokenized_splits = wiki.map(lambda ex: {'passages':[doc.split() for doc in ex['passages']]}, num_proc=num_proc)['passages']
    # print(len(tokenized_splits))
    # print(len(tokenized_splits[0]))
    # print(len(tokenized_splits[0][0]))
    # print(tokenized_splits[:10])
    tokenized_splits = [chunk for article in tokenized_splits for chunk in article]
    # print(len(tokenized_splits))
    # print(len(tokenized_splits[0]))
    # print(tokenized_splits[:10])
    del wiki
    # quit(0)

    # print(docs[:10])
    bm25 = BM25(tokenized_splits)

    train_questions = load_dataset('ContextualAI/nq', split='train')
    train_dataset = train_questions.map(add_retrieval_results, num_proc=num_proc)
    train_dataset = train_dataset.rename_columns({'query':'questions', 'gold_generation':'answers'})
    train_dataset = train_dataset.filter(lambda ex: len(ex['retrieved']) == retrieve_k)

    dev_questions = load_dataset('ContextualAI/nq', split='dev')
    dev_dataset = dev_questions.map(add_retrieval_results, num_proc=num_proc)
    dev_dataset = dev_dataset.rename_columns({'query':'questions', 'gold_generation':'answers'})
    dev_dataset = dev_dataset.filter(lambda ex: len(ex['retrieved']) == retrieve_k)

    test_questions = load_dataset('ContextualAI/nq', split='test')
    test_dataset = test_questions.map(add_retrieval_results, num_proc=num_proc)
    test_dataset = test_dataset.rename_columns({'query':'questions', 'gold_generation':'answers'})
    test_dataset = test_dataset.filter(lambda ex: len(ex['retrieved']) == retrieve_k)

    combined_dataset = DatasetDict({'train':train_dataset, 'dev':dev_dataset, 'test':test_dataset})
    combined_dataset.push_to_hub(f'ndc227/{args.output_name}', private=True)
    # quit(0)
elif OPTION == 2:
    docs = [DocWrapper(' '.join(ex['passages'])) for ex in tqdm(wiki)]
    del wiki

    rec_char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    rec_char_splitter_2 = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=200, add_start_index=True
    )
    token_splitter = TokenTextSplitter(
        encoding_name='gpt2', chunk_size=100, chunk_overlap=0
    )
    char_splitter = CharacterTextSplitter(
        separator='\n', is_separator_regex=False
    )
    nltk_splitter = NLTKTextSplitter(
        separator='\n\n', language='english'
    )
    sentence_transformer_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=50, model_name='sentence-transformers/all-mpnet-base-v2', tokens_per_chunk=None
    )
    spacy_splitter = SpacyTextSplitter(
        separator='\n\n', pipeline='en_core_web_sm', max_length=1000000
    )
    semantic_chunker = SemanticChunker(
        embeddings=HuggingFaceEmbeddings(), buffer_size=1, add_start_index=True
    )
    #... add more, like the SemanticChunker, etc.                    # DONE
    #... use different parameters for these (especially chunk sizes) # TODO: How wide of a parameter space should we use? Default is 4000/200

    all_splits, all_ids = [], []
    splitters = [rec_char_splitter, rec_char_splitter_2, token_splitter, char_splitter, nltk_splitter, \
                sentence_transformer_splitter, spacy_splitter, semantic_chunker]
    # splitters = [rec_char_splitter, token_splitter]
    for splitter in splitters:
        if splitter.__class__.__name__ != 'SemanticChunker':
            print(f'{splitter.__class__.__name__}_{splitter._chunk_size}_{splitter._chunk_overlap}')
        else:
            print(f'{splitter.__class__.__name__}')
        splits = get_splits(splitter, docs)
        all_splits.extend(splits)
        all_ids.extend([f'id_{splitter.__class__.__name__}_{ii}' for ii in range(len(splits))])

    # df = pd.DataFrame.from_dict({'chunk': all_splits, 'id': all_ids})
    # df.to_json('data/chunks.json')

    tokenized_splits = [x.split() for x in all_splits]
    bm25 = BM25(tokenized_splits)

    train_questions = load_dataset('ContextualAI/nq', split='train')
    train_dataset = train_questions.map(add_retrieval_results, num_proc=num_proc)
    train_dataset = train_dataset.rename_columns({'query':'questions', 'gold_generation':'answers'})
    train_dataset = train_dataset.filter(lambda ex: len(ex['retrieved']) == retrieve_k)

    dev_questions = load_dataset('ContextualAI/nq', split='dev')
    dev_dataset = dev_questions.map(add_retrieval_results, num_proc=num_proc)
    dev_dataset = dev_dataset.rename_columns({'query':'questions', 'gold_generation':'answers'})
    dev_dataset = dev_dataset.filter(lambda ex: len(ex['retrieved']) == retrieve_k)

    test_questions = load_dataset('ContextualAI/nq', split='test')
    test_dataset = test_questions.map(add_retrieval_results, num_proc=num_proc)
    test_dataset = test_dataset.rename_columns({'query':'questions', 'gold_generation':'answers'})
    test_dataset = test_dataset.filter(lambda ex: len(ex['retrieved']) == retrieve_k)

    combined_dataset = DatasetDict({'train':train_dataset, 'dev':dev_dataset, 'test':test_dataset})
    combined_dataset.push_to_hub(f'ndc227/{args.output_name}', private=True)