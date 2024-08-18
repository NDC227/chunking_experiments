import numpy as np
from transformers import AutoTokenizer
import wandb
from transformers import AutoModelForCausalLM, AutoModel
import torch
import torch.nn as nn
import lightning as L
from huggingface_hub import PyTorchModelHubMixin
import evaluate
from tqdm import tqdm

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
        x = self.embedding(input)
        x = self.encoding_block(x, src_key_padding_mask=mask)
        x = torch.mean(x, dim=1)
        return x

class ReplugTransformer(L.LightningModule, PyTorchModelHubMixin):
    def __init__(self, llm_model):
        super().__init__()
        self.model_id = llm_model
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir='/nlp/scr/ayc227/.cache/huggingface/models')
        self.llm_tokenizer.padding_side = 'left'
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.vocab_size = len(self.llm_tokenizer)
        self.reranker = AutoModel.from_pretrained('BAAI/bge-m3', cache_dir='/nlp/scr/ayc227/.cache/huggingface/models')
        self.reranker_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
        self.add_ids = False
        self.length_penalty = 0.0
        self.log_wandb = False

    def forward(self, questions, docs):
        inputs = self.reranker_tokenizer(questions, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
        scores = self.reranker(**inputs, return_dict=True)[0][:, 0]
        query_embeddings = torch.nn.functional.normalize(scores, p=2, dim=1)
        query_embeddings = query_embeddings.unsqueeze(1)
        # print(query_embeddings)
        docs = [d for doc in docs for d in doc]
        inputs = self.reranker_tokenizer(docs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
        scores = self.reranker(**inputs, return_dict=True)[0][:, 0]
        doc_embeddings = torch.nn.functional.normalize(scores, p=2, dim=1)
        _, L = doc_embeddings.shape
        doc_embeddings = torch.transpose(torch.reshape(doc_embeddings, (len(questions), -1, L)), 1, 2)
        # print(doc_embeddings.shape, doc_embeddings)
        # print('dims', query_embeddings.shape, doc_embeddings.shape)
        scores = torch.bmm(query_embeddings, doc_embeddings)
        scores = torch.squeeze(scores)
        # print('scores', scores)
        return scores
    
    def expand_data(self, data, batch_size, num_docs):                   # Data: B x S
        expanded_data = torch.unsqueeze(data, 1)                         # B x 1 x S
        expanded_data = expanded_data.expand(-1, num_docs, -1)           # B x K x S
        return torch.reshape(expanded_data, (batch_size * num_docs, -1)) # BK x S

    def training_step(self, batch, batch_idx):
        questions = batch['questions']
        docs = batch['new_chunks']
        # chunker_ids = batch['chunker_ids']
        # answers = batch['answers']
        
        self.reranker = self.reranker.to(device)
        self.reranker.train()
        print(1, torch.cuda.mem_get_info(), torch.cuda.memory_reserved(), torch.cuda.memory_allocated())

        # Run an LLM to get the NLLs (logits?)
        with torch.no_grad():
            llm_scores = torch.stack(batch['llm_scores'],dim=1)
            if self.length_penalty > 0:
                doc_lengths = torch.tensor([[len(d) for d in doc] for doc in docs]).transpose(0, 1).to(device)
                print('doc_lengths', doc_lengths.shape)
                llm_scores = llm_scores - doc_lengths * self.length_penalty
            llm_dist = torch.nn.functional.softmax(llm_scores, dim=1)
            # print('llm_scores', llm_scores.shape)
            # print('llm', llm_dist.shape, llm_dist)
        print(2, torch.cuda.mem_get_info(), torch.cuda.memory_reserved(), torch.cuda.memory_allocated())

        # Normalize the retrieval scores from the forward pass
        reranker_scores = self(questions, docs) # Reranker forward pass
        reranker_dist = torch.nn.functional.log_softmax(reranker_scores, dim=1)
        # print('rerank', reranker_dist.shape, reranker_dist)        
        print(3, torch.cuda.mem_get_info(), torch.cuda.memory_reserved(), torch.cuda.memory_allocated())

        # Compute loss = kldiv(scores, nlls)
        lossfn = torch.nn.KLDivLoss(reduction='batchmean')
        loss = lossfn(reranker_dist, llm_dist) # see docs for notation https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
        # print(loss)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(questions))
        if self.log_wandb:
            wandb.log({'train_loss': loss.item()})
        del reranker_scores, reranker_dist, llm_scores, llm_dist, lossfn
        torch.cuda.empty_cache()
        return loss

    def configure_optimizers(self):
        params = self.reranker.parameters()
        return torch.optim.AdamW(params, lr=1e-4)

    def validation_step(self, batch, batch_idx):
        pass

    def inference(self, questions, docs, top_k):
        scores = self.forward(questions, docs)
        order = torch.argsort(scores, descending=True)
        scores, _ = torch.sort(scores, descending=True)
        return scores[:,:top_k], order[:,:top_k]
    
    def ensemble_predict(self, questions, docs):
        if self.model_id.startswith('meta-llama'):
            query = [f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an assistant who gives short, succinct answers to questions. Please answer the following questions using the contexts given below: ''' + \
'\n'.join(['Context: ' + d for d in doc]) +\
f'''<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Question: who was leander paes partner in the mixed doubles at the us open in 2008?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Answer: Cara Black<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Question: who takes over after a president is impeached?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Answer: vice president<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Question: who plays the dogs voice in downward dog?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Answer: Samm Hodges<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Question: when did the name of persia change to iran?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Answer: 1935<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Question: the common name for a modulator demodulator is?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Answer: modem<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Question: what is the term for how steep a line is in math?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Answer: slope or gradient<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Question: how many ep are there in sacred games?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Answer: 8<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Question: neo malthusians believe that the solution to poverty is?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Answer: abstinence , delayed marriage<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Question: {questions[i]}?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Answer:''' for i, doc in enumerate(docs)]
            
        elif self.model_id.startswith('microsoft'):
            query = [f'''<|system|>
You are an assistant who gives short, succinct answers to questions. Please answer the following questions using the contexts given below: ''' + \
'\n'.join(['Context: ' + d for d in doc]) +\
f'''<|end|>
<|user|>
Question: who was leander paes partner in the mixed doubles at the us open in 2008?<|end|>
<|assistant|>
Answer: Cara Black<|end|>
<|user|>
Question: who takes over after a president is impeached?<|end|>
<|assistant|>
Answer: vice president<|end|>
<|user|>
Question: who plays the dogs voice in downward dog?<|end|>
<|assistant|>
Answer: Samm Hodges<|end|>
<|user|>
Question: when did the name of persia change to iran?<|end|>
<|assistant|>
Answer: 1935<|end|>
<|user|>
Question: {questions[i]}?<|end|>
<|assistant|>
Answer:''' for i, doc in enumerate(docs)]
            
        else:
            query = ['''You are an assistant who gives short, succinct answers to questions. Please answer the following questions using the contexts given below: '''+ \
'\n'.join(['Context: ' + d for d in doc]) + \

f'''
Question: who was leander paes partner in the mixed doubles at the us open in 2008?
Answer: Cara Black

Question: who takes over after a president is impeached?
Answer: vice president

Question: who plays the dogs voice in downward dog?
Answer: Samm Hodges

Question: when did the name of persia change to iran?
Answer: 1935

Question:{questions[i]}?
Answer:''' for i, doc in enumerate(docs)]
        tokenized_query = self.llm_tokenizer(query, padding=True, truncation=True, max_length=4000, return_tensors='pt').to(device)
        print(tokenized_query['input_ids'].shape, tokenized_query['input_ids'])
        print(torch.max(tokenized_query['input_ids']), torch.min(tokenized_query['input_ids']))
        print(2.1, torch.cuda.mem_get_info(), torch.cuda.memory_reserved(), torch.cuda.memory_allocated())
        outputs = self.llm.generate(**tokenized_query, max_new_tokens=16, tokenizer=self.llm_tokenizer, stop_strings=['\n'])
        print(2.2, torch.cuda.mem_get_info(), torch.cuda.memory_reserved(), torch.cuda.memory_allocated())
        outputs = outputs[:,len(tokenized_query['input_ids'][0]):]
        # print(outputs.shape)
        decoded_outputs = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_outputs = [output.strip() for output in decoded_outputs]
        # print('decoded_outputs', decoded_outputs)
        del tokenized_query, outputs
        torch.cuda.empty_cache()
        return decoded_outputs

    def evaluate(self, valid_loader, top_k, experiment):
        with torch.no_grad():
            if self.model_id.startswith('facebook'):
                self.llm = AutoModelForCausalLM.from_pretrained(self.model_id, cache_dir='/nlp/scr/ayc227/.cache/huggingface/models')
            else:
                self.llm = AutoModelForCausalLM.from_pretrained(self.model_id, cache_dir='/nlp/scr/ayc227/.cache/huggingface/models', 
                                                        torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2')
            self.llm.resize_token_embeddings(self.vocab_size)
            self.llm.to(device)
            self.llm.eval()

            em = evaluate.load('exact_match')
            predictions = []
            references = []
            
            for batch in tqdm(valid_loader):
                print(1, torch.cuda.mem_get_info(), torch.cuda.memory_reserved(), torch.cuda.memory_allocated())
                questions = batch['questions']
                if experiment == '1' or experiment == '1.5':
                    docs = batch['retrieved']
                    answers = batch['answers']
                else:
                    docs = batch['new_chunks']
                    chunker_ids = batch['chunker_ids']
                    answers = batch['answers']
                
                if experiment == '1':
                    new_docs = np.asarray(docs)[:top_k, :].T

                elif experiment == '1.5' or experiment == '2' or experiment == '3':
                    with torch.no_grad():
                        self.reranker.to(device)
                        self.reranker.eval()
                        docs = np.asarray(docs).T
                        new_docs = []
                        for i in range(len(questions)):
                            tokenized_question = self.reranker_tokenizer(questions[i], padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
                            query_scores = self.reranker(**tokenized_question, return_dict=True)[0][:, 0]
                            query_embeddings = torch.nn.functional.normalize(query_scores, p=2, dim=1)
                            
                            tokenized_docs = self.reranker_tokenizer(list(docs[i]), padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
                            docs_scores = self.reranker(**tokenized_docs, return_dict=True)[0][:, 0]
                            docs_embeddings = torch.nn.functional.normalize(docs_scores, p=2, dim=1)
                            
                            scores = (query_embeddings @ docs_embeddings.T)[0]
                            ranking = torch.argsort(scores, dim=0, descending=True)
                            top_k_docs = np.take(np.asarray(docs[i]), ranking[:top_k].cpu().numpy(), axis=0)
                            new_docs.append(list(top_k_docs))
                        del tokenized_question, tokenized_docs, query_scores, docs_scores, query_embeddings, docs_embeddings, scores, ranking

                elif experiment == '3.5':
                    scores = batch['llm_scores']
                    scores = torch.stack(scores).T
                    ranking = torch.argsort(torch.tensor(scores), descending=True, axis=1)
                    ranking = ranking[:,:top_k]
                    docs = np.asarray(docs).T
                    new_docs = np.take_along_axis(docs, ranking.numpy(), axis=1)

                elif experiment == '4':
                    tokenized_questions = self.llm_tokenizer(questions, padding=True, return_tensors='pt').to(device)
                    if self.add_ids:
                        tokenized_docs = self.llm_tokenizer([chunker_ids[i][j] + ' ' + x for i, retr in enumerate(docs) for j, x in enumerate(retr)], padding=True, return_tensors='pt').to(device)
                    else:
                        tokenized_docs = self.llm_tokenizer([x for retr in docs for x in retr], padding=True, truncation=True, max_length=400, return_tensors='pt').to(device)
                    tokenized_docs['input_ids'] = torch.transpose(torch.reshape(tokenized_docs['input_ids'], (len(docs), len(docs[0]), -1)), 0, 1)
                    tokenized_docs['attention_mask'] = torch.transpose(torch.reshape(tokenized_docs['attention_mask'], (len(docs), len(docs[0]), -1)), 0, 1)

                    rerank_scores, rerank_order = self.inference(tokenized_questions, tokenized_docs, top_k)
                    # print('rerank_scores', rerank_scores.shape, rerank_scores)
                    docs = np.asarray(docs).T
                    new_docs = np.take_along_axis(np.asarray(docs), rerank_order.cpu().numpy(), axis=1)
                    del tokenized_questions, tokenized_docs, rerank_scores, rerank_order
                    torch.cuda.empty_cache()
                    
                print(2, torch.cuda.mem_get_info(), torch.cuda.memory_reserved(), torch.cuda.memory_allocated())

                preds = self.ensemble_predict(questions, new_docs)
                print(3, torch.cuda.mem_get_info(), torch.cuda.memory_reserved(), torch.cuda.memory_allocated())

                predictions += preds
                references += [[answer] for answer in answers]
                print(preds, answers)

            em_results = em.compute(predictions=predictions, references=list(np.asarray(references).flatten()), ignore_case=True, ignore_punctuation=True)
            print(em_results)