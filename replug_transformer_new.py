import numpy as np
from transformers import AutoTokenizer
import wandb
from transformers import AutoModelForCausalLM
import torch
import torch.nn as nn
import lightning as L
from huggingface_hub import PyTorchModelHubMixin
import evaluate
from tqdm import tqdm

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = 'cuda' if torch.cuda.is_available else 'cpu'

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
        # print(input.device)
        # print(next(self.embedding.parameters()).device)
        x = self.embedding(input)
        x = self.encoding_block(x, src_key_padding_mask=mask)
        # x = self.encoding_block(x)
        x = torch.mean(x, dim=1)
        return x

class ReplugTransformer(L.LightningModule, PyTorchModelHubMixin):
    def __init__(self, llm_model):
        super().__init__()
        # self.encoder = Transformer(vocab_size=vocab_size)
        self.model_id = llm_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir='/nlp/scr/ayc227/.cache/huggingface/models')
        self.tokenizer.padding_side = 'left'
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.vocab_size = len(self.tokenizer)
        self.question_encoder = Encoder(self.vocab_size)
        self.docs_encoder = Encoder(self.vocab_size)
    
    def forward(self, questions, docs):
        print(1.2, torch.cuda.mem_get_info(), torch.cuda.memory_reserved(), torch.cuda.memory_allocated())
        questions_input, questions_mask = questions['input_ids'], torch.logical_not(questions['attention_mask'].to(dtype=torch.bool))
        docs_input, docs_mask = docs['input_ids'], torch.logical_not(docs['attention_mask'].to(dtype=torch.bool))
        # Encode questions and documents
        # print('questions', questions)
        # print(len(docs), questions.shape, docs[0].shape)
        # print('docs', docs_input.shape, docs_input)
        n_examples = questions_input.shape[0]
        q_emb = self.question_encoder(questions_input, questions_mask)
        print(1.3, torch.cuda.mem_get_info(), torch.cuda.memory_reserved(),  torch.cuda.memory_allocated())
        # print('q_emb', q_emb.shape, q_emb)
        # Squeeze and unsqueeze to pass batch of retrievals into encoder at once?
        # Turn B x K x L to (BK) x L
        B, K, L = docs_input.shape
        # print(B, K, L)
        squeeze_docs_input = torch.reshape(docs_input, (B * K, -1))
        squeeze_docs_mask = torch.reshape(docs_mask, (B * K, -1))
        print('1.3.0', torch.cuda.mem_get_info(), torch.cuda.memory_reserved(),  torch.cuda.memory_allocated())
        # print(squeeze_docs_input.shape)
        d_embs = self.docs_encoder(squeeze_docs_input, squeeze_docs_mask)
        d_embs = torch.reshape(d_embs, (B, K, -1))
        print('1.3.1 docs_encoder input size:', squeeze_docs_input.element_size()*squeeze_docs_input.nelement())
        # print(d_embs.shape)
        print(1.4, torch.cuda.mem_get_info(), torch.cuda.memory_reserved(),  torch.cuda.memory_allocated())

        d_scores = torch.einsum('bij,bjk->bik', d_embs, torch.unsqueeze(q_emb, -1))
        d_scores = d_scores.squeeze()
        print(1.5, torch.cuda.mem_get_info(), torch.cuda.memory_reserved(),  torch.cuda.memory_allocated())

        del questions_input, questions_mask, docs_input, docs_mask, squeeze_docs_input, squeeze_docs_mask, q_emb, d_embs
        torch.cuda.empty_cache()
        # print('d_scores', d_scores.shape, d_scores)
        print(1.6, torch.cuda.mem_get_info(), torch.cuda.memory_reserved(),  torch.cuda.memory_allocated())

        return d_scores
    
    def expand_data(self, data, batch_size, num_docs):                   # Data: B x S
        expanded_data = torch.unsqueeze(data, 1)                         # B x 1 x S
        expanded_data = expanded_data.expand(-1, num_docs, -1)           # B x K x S
        return torch.reshape(expanded_data, (batch_size * num_docs, -1)) # BK x S

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        opt = self.optimizers()
        print(1, torch.cuda.mem_get_info(), torch.cuda.memory_reserved(), torch.cuda.memory_allocated())
        # TODO: Make this actually work with the dataset constructed above
        # Pre-Process data (moved from above in order to batch tokenize for efficiency)
        questions = batch['questions']
        docs = batch['new_chunks']
        answers = batch['answers']
        # print('questions', questions)
        # print('answers', answers)
        tokenized_questions = self.tokenizer(questions, padding=True, return_tensors='pt').to(device)
        # tokenized_docs = self.tokenizer([x for retr in docs for x in retr], padding='max_length', truncation=True, max_length=100, return_tensors='pt').to(device)
        tokenized_docs = self.tokenizer([x for retr in docs for x in retr], padding=True, return_tensors='pt').to(device)
        tokenized_docs['input_ids'] = torch.transpose(torch.reshape(tokenized_docs['input_ids'], (len(docs), len(docs[0]), -1)), 0, 1)
        tokenized_docs['attention_mask'] = torch.transpose(torch.reshape(tokenized_docs['attention_mask'], (len(docs), len(docs[0]), -1)), 0, 1)
        tokenized_answers = self.tokenizer(answers, padding=True, return_tensors='pt').to(device)

        self.question_encoder = self.question_encoder.to(device)
        self.docs_encoder = self.docs_encoder.to(device)
        print(1.01, torch.cuda.mem_get_info(), torch.cuda.memory_reserved(), torch.cuda.memory_allocated())

        print("1.1 size of inputs:", tokenized_questions['input_ids'].element_size()*tokenized_questions['input_ids'].nelement() + \
              tokenized_docs['input_ids'].element_size()*tokenized_docs['input_ids'].nelement() + \
              tokenized_answers['input_ids'].element_size()*tokenized_answers['input_ids'].nelement())


        # Run an LLM to get the NLLs (logits?)
        # llm_output = self.llm_pass(tokenized_questions, tokenized_docs, tokenized_answers)
        # llm_dist = torch.nn.functional.softmax(llm_output, dim=1)
        # print(batch['llm_scores'])
        llm_scores = torch.stack(batch['llm_scores'],dim=1)
        # print('llm_scores', llm_scores.shape)
        llm_dist = torch.nn.functional.softmax(llm_scores, dim=1)
        # print('llm', llm_dist.shape, llm_dist)
        print(2, torch.cuda.mem_get_info(), torch.cuda.memory_reserved(), torch.cuda.memory_allocated())
        # quit(0)
        torch.cuda.empty_cache()
        print(3, torch.cuda.mem_get_info(), torch.cuda.memory_reserved(), torch.cuda.memory_allocated())

         # Normalize the retrieval scores from the forward pass
        reranker_output = self(tokenized_questions, tokenized_docs)  # output is retrieval scores?
        rerank_dist = torch.nn.functional.log_softmax(reranker_output, dim=1)
        del tokenized_questions, tokenized_docs, reranker_output
        
        print(4, torch.cuda.mem_get_info(), torch.cuda.memory_reserved(), torch.cuda.memory_allocated())

        # print('rerank', rerank_dist.shape, rerank_dist)
        # print(rerank_dist)
        # print('llm', llm_dist.shape, llm_dist)
        # print(llm_dist)
        # quit(0)

        # Compute loss = kldiv(scores, nlls)
        lossfn = torch.nn.KLDivLoss(reduction='batchmean')
        loss = lossfn(rerank_dist, llm_dist) # see docs for notation https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
        # print(loss)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(questions))
        # wandb.log({'train_loss': loss.item()})
        # opt.zero_grad()
        del tokenized_answers, rerank_dist, llm_dist, lossfn
        torch.cuda.empty_cache()
        return loss

    def configure_optimizers(self):
        params = list(self.question_encoder.parameters()) + list(self.docs_encoder.parameters())
        return torch.optim.AdamW(params, lr=1e-4)

    def validation_step(self, batch, batch_idx):
        pass