import numpy as np
import scipy
import pandas as pd
from datasets import load_dataset
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter, CharacterTextSplitter, NLTKTextSplitter,\
                                     SentenceTransformersTokenTextSplitter, SpacyTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer
import wandb
from transformers import AutoModelForCausalLM, AutoModel
import torch
import torch.nn as nn
import lightning as L
from tqdm import tqdm

from fast_bm25 import BM25

device = "cuda" if torch.cuda.is_available() else "cpu"

class DocWrapper(object):
  def __init__(self, s):
    self.page_content = s['contexts']
    self.metadata = {}
ds = load_dataset('cjlovering/natural-questions-short')
docs = [DocWrapper(ex) for ex in ds['train']]
# print(docs[:10])
# print(docs[0].page_content)

if not os.path.isfile('data/chunks.json'):
    def get_splits(splitter, data):
        return [split.page_content for split in splitter.split_documents(docs)]

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
                sentence_transformer_splitter]
    splitters = [rec_char_splitter, token_splitter]
    for splitter in splitters:
        if splitter.__class__.__name__ != 'SemanticChunker':
            print(f'{splitter.__class__.__name__}_{splitter._chunk_size}_{splitter._chunk_overlap}')
        else:
            print(f'{splitter.__class__.__name__}')
        splits = get_splits(splitter, docs)
        all_splits.extend(splits)
        all_ids.extend([f'id_{splitter.__class__.__name__}_{ii}' for ii in range(len(splits))])
    
    df = pd.DataFrame.from_dict({'chunk': all_splits, 'id': all_ids})
    df.to_json('data/chunks.json')
else:
    df = pd.read_json('data/chunks.json')
    all_splits = list(df['chunk'].values)
    all_ids = list(df['id'].values)

tokenized_splits = [x.split() for x in all_splits]
bm25 = BM25(tokenized_splits)

def retrieve_from_query(query, k):
    tokenized_query = query.split()
    retrieve = bm25.get_top_n(tokenized_query, tokenized_splits, n=k)
    retrieved_docs = [' '.join(x) for x in retrieve]
    return retrieved_docs

batch_size = 4
model_id = 'facebook/opt-125m'
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Now pass the retrieval results to the LLM (basically, RAG with frozen components)

# Baseline: deduplicate retrieval results somehow

# The neural net we train:
# - For each question:
#    - Encode each chunk and output a score (how to encode?)
#    - Turn all scores over all chunks into a distribution P(d|q)
#    - For each chunk get the NLL of the correct answer as another distribution Q(d|q) (from a RAG 'open domain QA' dataset; use HF transformers to get the NLL)
#    - Minimize the KL div of those two distributions (KL_div(P||Q)) (https://arxiv.org/abs/2301.12652 - REPLUG)
# - Add loss term for the length of the sequence (number of tokens)
# - Train

# For (fast) BM25
def add_retrieval_results(ex):
    ex['retrieved'] = retrieve_from_query(ex['questions'][0]['input_text'], k=10)
    # one alternative here for speed could be to not use Chroma but some kind of sparse search like Elastic
    return ex

# Data pre-processing (tokenize and de-nesting)
def preprocess(ex):
    ex['questions'] = ex['questions'][0]['input_text']
    ex['answers'] = ex['answers'][0]['span_text']
    return ex

tiny_train_dataset = Dataset.from_dict(ds['train'][:])
tiny_train_dataset = tiny_train_dataset.map(add_retrieval_results)
tiny_train_dataset = tiny_train_dataset.map(preprocess)
tiny_train_dataset = tiny_train_dataset.remove_columns(['contexts', 'has_correct_context', 'name', 'id'])
tiny_train_loader = DataLoader(tiny_train_dataset, batch_size=batch_size, shuffle=True)

tiny_valid_dataset = Dataset.from_dict(ds['validation'][:])
tiny_valid_dataset = tiny_valid_dataset.map(add_retrieval_results)
tiny_valid_dataset = tiny_valid_dataset.map(preprocess)
tiny_valid_loader = DataLoader(tiny_valid_dataset, batch_size=batch_size, shuffle=True)

# train_dataset = ds['train'].map(add_retrieval_results)
# train_dataset = train_dataset.map(tokenize)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# valid_dataset = ds['valid'].map(add_retrieval_results)
# valid_dataset = valid_dataset.map(tokenize)
# valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
wandb.login(key="fc6c9280f011612e6aeb6c45fd6f79f7d08c56dc")
wandb.init(
    # set the wandb project where this run will be logged
    entity="ndc227-stanford-university",
    project='chunking_experiments',

    # track hyperparameters and run metadata
    config={
    'learning_rate': 0.1,
    'architecture': 'Transformer',
    'dataset': 'NQ',
    'epochs': 10,
    }
)

llm_model = AutoModelForCausalLM.from_pretrained(model_id)

class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = 256
        self.dim_ff = 1024
        self.n_head = 4
        self.n_layers = 4
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.emb_dim)
        self.encoding_layer = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=self.n_head, dim_feedforward=self.dim_ff, batch_first=True)
        self.encoding_block = nn.TransformerEncoder(self.encoding_layer, num_layers=self.n_layers)

    def forward(self, input, mask):
        print(input.device)
        print(next(self.embedding.parameters()).device)
        x = self.embedding(input)
        x = self.encoding_block(x, src_key_padding_mask=mask)
        # x = self.encoding_block(x)
        x = torch.mean(x, dim=1)
        return x

# TODO by Andrew: fill in the rest from here: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
class ReplugTransformer(L.LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        # self.encoder = Transformer(vocab_size=vocab_size)
        self.encoder = Encoder(vocab_size)
        self.llm = llm_model  # LLM to get NLLs for reference distribution in KL div #TODO: separate 
        for param in self.llm.parameters():
            param.requires_grad = False
    
    def forward(self, questions, docs):
        questions, questions_mask = questions['input_ids'], torch.logical_not(questions['attention_mask'].to(dtype=torch.bool))
        docs, docs_mask = docs['input_ids'], torch.logical_not(docs['attention_mask'].to(dtype=torch.bool))
        # Encode questions and documents
        print('questions', questions)
        # print(len(docs), questions.shape, docs[0].shape)
        print('docs', docs.shape, docs)
        n_examples = questions.shape[0]
        q_emb = self.encoder(questions, questions_mask)
        print('q_emb', q_emb.shape, q_emb)
        # Squeeze and unsqueeze to pass batch of retrievals into encoder at once?
        # Turn B x K x L to (BK) x L
        B, K, L = docs.shape
        print(B, K, L)
        input_docs = torch.reshape(docs, (B * K, L))
        input_docs_mask = torch.reshape(docs_mask, (B*K, -1))
        print(input_docs.shape)
        d_embs = self.encoder(input_docs, input_docs_mask)
        d_embs = torch.reshape(d_embs, (B, K, -1))
        print(d_embs.shape)

        d_scores = torch.einsum('bij,bjk->bik', d_embs, torch.unsqueeze(q_emb, -1))
        d_scores = d_scores.squeeze()

        del q_emb
        del d_embs
        torch.cuda.empty_cache()
        print('d_scores', d_scores.shape, d_scores)

        return d_scores
    
    def expand_data(self, data, batch_size, num_docs):                   # Data: B x S
        expanded_data = torch.unsqueeze(data, 1)                         # B x 1 x S
        expanded_data = expanded_data.expand(-1, num_docs, -1)           # B x K x S
        return torch.reshape(expanded_data, (batch_size * num_docs, -1)) # BK x S

    def new_llm_pass(self, questions, docs, answers):
        # Again, reshape to put into LLM as B x ? shape
        questions, questions_mask = questions['input_ids'], torch.logical_not(questions['attention_mask'].to(dtype=torch.bool))
        docs, docs_mask = docs['input_ids'], torch.logical_not(docs['attention_mask'].to(dtype=torch.bool))
        print(answers['attention_mask'])
        print(answers['attention_mask'].to(dtype=torch.bool))
        print(torch.logical_not(answers['attention_mask'].to(dtype=torch.bool)))
        answers, answers_mask = answers['input_ids'], answers['attention_mask'].to(dtype=torch.bool) # B x A
        
        print(answers_mask)
        B, K, L = docs.shape
        docs = torch.reshape(docs, (B*K, L))                   # BK x L
        docs_mask = torch.reshape(docs_mask, (B*K, L))         # BK x L
        _, answer_length = answers.shape
        expanded_questions = self.expand_data(questions, B, K)
        expanded_questions_mask = self.expand_data(questions_mask, B, K)
        # print('expanded_questions', expanded_questions.shape, expanded_questions)
        # TODO: reconsider how the inputs are combined, right now they are just concatenated with the padding... Retokenize?
        combined_input = torch.cat([docs, expanded_questions], dim=1)      # BK x (S + L)
        combined_input_mask = torch.cat([docs_mask, expanded_questions_mask], dim=1) # BK x (S + L)
        # print('combined_input', combined_input.shape)
        expanded_answers = self.expand_data(answers, B, K)
        expanded_answers_mask = self.expand_data(answers_mask, B, K)
        print("expanded_answers_mask", expanded_answers_mask)
        all_scores = []
        for i in range(answer_length):
            expected_tokens = expanded_answers[:, i]             # BK x 1
            expected_tokens_mask = expanded_answers_mask[:, i]   # BK x 1
            # print('expected_tokens', expected_tokens.shape, expected_tokens)
            outputs = self.llm(combined_input, combined_input_mask)                   # BK x (S+L+i) x V
            # print('outputs[logits]', outputs['logits'].shape)
            last_outputs = outputs['logits'][:, -1, :]           # BK x V
            # print('last_outputs', last_outputs.shape, last_outputs)
            scores = last_outputs[torch.arange(B*K), expected_tokens] # BK x 1
            # print('scores', scores.shape, scores)
            # print("expected_tokens_mask", expected_tokens_mask)
            scores = torch.where(expected_tokens_mask == 1, scores, torch.nan)
            # print('scores', scores.shape, scores)
            all_scores.append(scores)

            combined_input = torch.cat([combined_input, torch.unsqueeze(expected_tokens, 1)], dim=1) #BK x (S + L + i)
            combined_input_mask = torch.cat([combined_input_mask, torch.unsqueeze(expected_tokens_mask, 1)], dim=1)
            del expected_tokens
            del expected_tokens_mask
            del outputs
            del last_outputs
            torch.cuda.empty_cache()
        all_scores = torch.stack([x for x in all_scores], dim=1)
        print('all_scores', all_scores.shape, all_scores)
        all_scores = all_scores.reshape(B, K, -1)
        print('all_scores', all_scores.shape, all_scores)
        all_scores = torch.nanmean(all_scores, dim=-1)
        print('all_scores', all_scores.shape, all_scores)
        return all_scores

    def training_step(self, batch, batch_idx):
        # TODO: Make this actually work with the dataset constructed above
        # Pre-Process data (moved from above in order to batch tokenize for efficiency)
        questions = batch['questions']
        docs = batch['retrieved']
        answers = batch['answers']
        tokenized_questions = tokenizer(questions, padding=True, return_tensors='pt').to(device)
        tokenized_docs = tokenizer([x for retr in docs for x in retr], padding='max_length', truncation=True, max_length=100, return_tensors='pt').to(device)
        tokenized_docs['input_ids'] = torch.reshape(tokenized_docs['input_ids'], (len(docs[0]), len(docs), -1))
        tokenized_docs['attention_mask'] = torch.reshape(tokenized_docs['attention_mask'], (len(batch['retrieved'][0]), len(batch['retrieved']), -1))
        tokenized_answers = tokenizer(answers, padding=True, return_tensors='pt').to(device)

        # Normalize the retrieval scores from the forward pass
        # TODO: Apply the masks in the scoring process - currently no masks but still converges on single batch

        reranker_output = self(tokenized_questions, tokenized_docs)  # output is retrieval scores?
        rerank_dist = torch.nn.functional.log_softmax(reranker_output, dim=1)
        
        # Run an LLM to get the NLLs - NLLs or is it just the logits??
        llm_output = self.new_llm_pass(tokenized_questions, tokenized_docs, tokenized_answers)
        llm_dist = torch.nn.functional.softmax(llm_output, dim=1)

        print('rerank', rerank_dist.shape, rerank_dist)
        # print(rerank_dist)
        print('llm', llm_dist.shape, llm_dist)
        # print(llm_dist)

        # Compute loss = kldiv(scores, nlls)
        lossfn = torch.nn.KLDivLoss(reduction='batchmean')
        loss = lossfn(rerank_dist, llm_dist) # see docs for notation https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
        print(loss)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        wandb.log({'train_loss': loss})
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.encoder.parameters(), lr=1e-4)
    
model = ReplugTransformer(vocab_size=tokenizer.vocab_size)
trainer = L.Trainer(max_epochs=1)
trainer.fit(model, tiny_train_loader, tiny_valid_loader)

#for batch in ds['train']:
#  batch_scores = net(batch) # Bx1 float
#  retrieval_scores = F.softmax(batch_scores)
#  nll_scores = llm(batch).mean() # Bx1 float
#  loss = kldiv(batch_scores, nll_scores)
#  # TODO: loss += weight_coeff * sum([len(x) for x in batch])
#  loss.backward()

# Process:
# - Retrieve with high n_results
# - Rerank using the above neural net
# - Take the top-k from that reranker
# - Give to LLM
# - Evaluate

# Once the above actually works:
# - Add additional deduplication steps?
# - Add a diversity term to the loss?
# - Add metadata to the chunk (e.g. prefix 'TokenTextSplitter' to each such chunk for the net() call, NOT the llm() call)


# Document score side:
# Input: BxS
# After retrieval: (BxS, BxKxL) (assume tokenized)
# After encoding: (BxSxE -> BxE), (BxKxE) (where E is embedding/hidden dim, separate encoders)
# After scoring: BxK (do bmm/einsum to get this)
# After normalize: BxK (logsoftmax, make sure you don't do double log)

# NLL score side:
# Input: BxS
# After encoding: Bx(S+L) -> Bx(Correct Answer)xV -> select logit of correct token, take mean -> Bx1

# 'Context: The largest city in Japan is Tokyo. Question: What is the largest city in Japan? Answer: ' -> what is the logit for Tokyo?
# Autoregressively generate for max_length_of_correct_answers_in_batch, giving you some matrix BxM,
# zero out all the components that you don't need (ie if the correct answer was 1 token, all the other tokens you can zero out),
# then take the mean of the non-zero elements to get Bx1 NLL tensor

# ^ TODO: double check that this is how Replug does it

# After loss: Bx1 (kldiv)

# Experimental protocol:
# 1. Validate that you get the above shapes
# 2. Print the BxK scores over time, make sure they change
# 3. 'Spike' the retrievals with 7 bad examples and 1 correct one, make sure the mass shifts to the correct one
# 4. Make sure train loss goes down (first on the spiked one, then on single batch, then all batches)
# 5. Run with fast BM25



