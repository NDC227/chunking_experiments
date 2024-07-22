import argparse
from datasets import load_dataset, Dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter, CharacterTextSplitter, NLTKTextSplitter,\
                                     SentenceTransformersTokenTextSplitter, SpacyTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from fast_bm25 import BM25

# Sample use:
# python create_datasets.py --dir data --output_name chunks_retrieve_100 --retrieve_k 100
argp = argparse.ArgumentParser()
argp.add_argument('--dir', default='data')
argp.add_argument('--output_name', default='chunks')
argp.add_argument('--retrieve_k', default=10, type=int)
args = argp.parse_args()

class DocWrapper(object):
  def __init__(self, s):
    self.page_content = s['contexts']
    self.metadata = {}

def get_splits(splitter, data):
    return [split.page_content for split in splitter.split_documents(docs)]

def retrieve_from_query(query, k):
    tokenized_query = query.split()
    retrieve = bm25.get_top_n(tokenized_query, tokenized_splits, n=k)
    retrieved_docs = [' '.join(x) for x in retrieve]
    return retrieved_docs

# For (fast) BM25
def add_retrieval_results(ex):
    ex['retrieved'] = retrieve_from_query(ex['questions'][0]['input_text'], k=args.retrieve_k)
    # one alternative here for speed could be to not use Chroma but some kind of sparse search like Elastic
    return ex

# Data pre-processing (tokenize and de-nesting)
def preprocess(ex):
    ex['questions'] = ex['questions'][0]['input_text']
    ex['answers'] = ex['answers'][0]['span_text']
    return ex

ds = load_dataset('cjlovering/natural-questions-short')
docs = [DocWrapper(ex) for ex in ds['train']]

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

train_dataset = Dataset.from_dict(ds['train'][:])
train_dataset = train_dataset.map(add_retrieval_results)
train_dataset = train_dataset.map(preprocess)
train_dataset = train_dataset.remove_columns(['contexts', 'has_correct_context', 'name', 'id'])

valid_dataset = Dataset.from_dict(ds['validation'][:])
valid_dataset = valid_dataset.map(add_retrieval_results)
valid_dataset = valid_dataset.map(preprocess)
valid_dataset = valid_dataset.remove_columns(['contexts', 'has_correct_context', 'name', 'id'])

train_dataset.to_json(f'{args.dir}/{args.output_name}_train.json')
valid_dataset.to_json(f'{args.dir}/{args.output_name}_valid.json')