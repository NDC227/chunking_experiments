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
        self.llm = AutoModelForCausalLM.from_pretrained(self.model_id, cache_dir='/nlp/scr/ayc227/.cache/huggingface/models')  # LLM to get NLLs for reference distribution in KL div 
        self.llm.resize_token_embeddings(self.vocab_size)
        for param in self.llm.parameters():
            param.requires_grad = False
    
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

    def llm_pass(self, questions, docs, answers):
        with torch.no_grad():
            # Again, reshape to put into LLM as B x ? shape
            questions_input, questions_mask = questions['input_ids'], questions['attention_mask'].to(dtype=torch.bool)
            docs_input, docs_mask = docs['input_ids'], docs['attention_mask'].to(dtype=torch.bool)
            # print(answers['attention_mask'])
            # print(answers['attention_mask'].to(dtype=torch.bool))
            # print(torch.logical_not(answers['attention_mask'].to(dtype=torch.bool)))
            answers_input, answers_mask = answers['input_ids'], answers['attention_mask'].to(dtype=torch.bool) # B x A
            
            # print(answers_mask)
            B, K, L = docs_input.shape
            docs_input = torch.reshape(docs_input, (B*K, L))                   # BK x L
            docs_mask = torch.reshape(docs_mask, (B*K, L))         # BK x L
            _, answer_length = answers_input.shape
            expanded_questions_input = self.expand_data(questions_input, B, K)
            expanded_questions_mask = self.expand_data(questions_mask, B, K)
            # print('expanded_questions', expanded_questions.shape, expanded_questions)
            # TODO: reconsider how the inputs are combined, right now they are just concatenated with the padding... Retokenize?
            combined_input = torch.cat([docs_input, expanded_questions_input], dim=1)      # BK x (S + L)
            combined_mask = torch.cat([docs_mask, expanded_questions_mask], dim=1) # BK x (S + L)
            # print('combined_input', combined_input.shape)
            expanded_answers_input = self.expand_data(answers_input, B, K)
            expanded_answers_mask = self.expand_data(answers_mask, B, K)
            # print('expanded_answers_mask', expanded_answers_mask)
            all_scores = []
            for i in range(answer_length):
                expected_tokens = expanded_answers_input[:, i]             # BK x 1
                expected_mask = expanded_answers_mask[:, i]   # BK x 1
                # print('expected_tokens', expected_tokens.shape, expected_tokens)
                outputs = self.llm(combined_input, combined_mask)                   # BK x (S+L+i) x V
                # print('outputs[logits]', outputs['logits'].shape)
                last_outputs = outputs['logits'][:, -1, :]           # BK x V
                # print('last_outputs', last_outputs.shape, last_outputs)
                scores = last_outputs[torch.arange(B*K), expected_tokens] # BK x 1
                # print('scores', scores.shape, scores)
                # print('expected_tokens_mask', expected_tokens_mask)
                scores = torch.where(expected_mask == 1, scores, torch.nan)
                # print('scores', scores.shape, scores)
                all_scores.append(scores)

                combined_input = torch.cat([combined_input, torch.unsqueeze(expected_tokens, 1)], dim=1) #BK x (S + L + i)
                combined_mask = torch.cat([combined_mask, torch.unsqueeze(expected_mask, 1)], dim=1)
                del expected_tokens, expected_mask, outputs, last_outputs
                torch.cuda.empty_cache()
            all_scores = torch.stack([x for x in all_scores], dim=1)
            # print('all_scores', all_scores.shape, all_scores)
            all_scores = all_scores.reshape(B, K, -1)
            # print('all_scores', all_scores.shape, all_scores)
            all_scores = torch.nanmean(all_scores, dim=-1)
            # print('all_scores', all_scores.shape, all_scores)
            del questions_input, questions_mask, docs_input, docs_mask, answers_input, answers_mask
            del expanded_questions_input, expanded_questions_mask, expanded_answers_input, expanded_answers_mask, combined_input, combined_mask
            torch.cuda.empty_cache()
            return all_scores
    
    def llm_pass_2(self, questions, docs, answers):
        with torch.no_grad():
            docs = np.asarray(docs).T
            B, K = len(docs), len(docs[0])
            # print(B, K)
            if self.model_id.startswith('meta-llama'):
                queries = [f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an assistant who gives short, succinct answers to questions. Please answer according to the following examples:<|eot_id|>

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
For the final question, use the following context to answer the question.
Context: {d}
Question: {questions[i]}?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Answer:""" for i, doc in enumerate(docs) for d in doc]
                
            elif self.model_id.startswith('microsoft'):
                queries = [f"""<|system|>
You are an assistant who gives short, succinct answers to questions. Please answer according to the following examples:<|end|>
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
For the final question, use the following context to answer the question.
Context: {d}
Question: {questions[i]}?<|end|>
<|assistant|>
Answer:""" for i, doc in enumerate(docs) for d in doc]
                
            else:
                queries = ["""Question: who was leander paes partner in the mixed doubles at the us open in 2008?
Answer: Cara Black

Question: who takes over after a president is impeached?
Answer: vice president
                           
Question: who plays the dogs voice in downward dog?
Answer: Samm Hodges
                           
Question: when did the name of persia change to iran?
Answer: 1935
""" + \
f'For the final question, answer using the given context:\n' + \
f'Context: {d}' + \
f'\nQuestion: {questions[i]}?\nAnswer:' for i, doc in enumerate(docs) for d in doc]
            
            # queries = [['Please answer the following question using the following context. The answer should not restate the question.\n\nContext: ' + d + '\n\nQuestion: ' + questions[i] + '?\n\nVery short answer:' for d in doc] for i, doc in enumerate(docs)]
            # queries = [q for query in queries for q in query]
            # print(queries[:6])
            # quit(0)
            tokenized_queries = self.tokenizer(queries, padding=True, return_tensors='pt').to(device)
            answers_input, answers_mask = answers['input_ids'], answers['attention_mask'].to(dtype=torch.bool) # B x A
            # print('answers_input', answers_input.shape, answers_input)
            print(3.1, torch.cuda.mem_get_info(), torch.cuda.memory_reserved(), torch.cuda.memory_allocated())

            _, answer_length = answers_input.shape
            combined_input = tokenized_queries['input_ids']      # BK x (S + L)
            combined_mask = tokenized_queries['attention_mask'] # BK x (S + L)
            # print('combined_input', combined_input.shape)
            expanded_answers_input = self.expand_data(answers_input, B, K)
            expanded_answers_mask = self.expand_data(answers_mask, B, K)
            # print('expanded_answers_mask', expanded_answers_mask)
            all_scores = []
            for i in range(answer_length):
                print(3.2, torch.cuda.mem_get_info(), i, torch.cuda.memory_reserved(), torch.cuda.memory_allocated())
                # print('combined_input', combined_input.shape)
                expected_tokens = expanded_answers_input[:, i]             # BK x 1
                expected_mask = expanded_answers_mask[:, i]   # BK x 1
                # print('expected_tokens', expected_tokens.shape, expected_tokens)

                # outputs = self.llm(input_ids=combined_input, attention_mask=combined_mask)                   # BK x (S+L+i) x V
                #     # print('outputs[logits]', outputs['logits'].shape)
                # last_outputs = outputs['logits'][:, -1, :]           # BK x V
                # # print('last_outputs', last_outputs.shape, last_outputs)
                # scores = last_outputs[torch.arange(B*K), expected_tokens] # BK x 1
                # all_scores.append(scores)

                # combined_input = torch.cat([combined_input, torch.unsqueeze(expected_tokens, 1)], dim=1) #BK x (S + L + i)
                # combined_mask = torch.cat([combined_mask, torch.unsqueeze(expected_mask, 1)], dim=1)
                # del expected_tokens, expected_mask, outputs, last_outputs
                # torch.cuda.empty_cache()
                
                scores = []
                split_factor = 10
                for j in range(split_factor):
                    # print(combined_input.shape, combined_input)
                    # print(B*K//10*j, B*K//10*(j+1))
                    # print(combined_input[B*K//10*j:B*K//10*(j+1),:].shape, combined_input[B*K//10*j:B*K//10*(j+1),:])
                    # print(combined_mask[B*K//10*j:B*K//10*(j+1),:].shape, combined_mask[B*K//10*j:B*K//10*(j+1),:])
                    outputs = self.llm(input_ids=combined_input[B*K//split_factor*j:B*K//split_factor*(j+1),:], attention_mask=combined_mask[B*K//split_factor*j:B*K//split_factor*(j+1),:])                   # BK x (S+L+i) x V
                    # print('outputs[logits]', outputs['logits'].shape)
                    last_outputs = outputs['logits'][:, -1, :]           # BK x V
                    # print('last_outputs', last_outputs.shape, last_outputs)
                    scores.append(last_outputs[torch.arange(B*K//split_factor), expected_tokens[B*K//split_factor*j:B*K//split_factor*(j+1)]]) # BK x 1
                    del outputs, last_outputs
                    torch.cuda.empty_cache()
                scores = torch.cat(scores, dim=0)
                # print('scores', scores.shape, scores)
                # print('expected_tokens_mask', expected_tokens_mask)
                scores = torch.where(expected_mask == 1, scores, torch.nan)
                # print('scores', scores.shape, scores)
                all_scores.append(scores)

                combined_input = torch.cat([combined_input, torch.unsqueeze(expected_tokens, 1)], dim=1) #BK x (S + L + i)
                combined_mask = torch.cat([combined_mask, torch.unsqueeze(expected_mask, 1)], dim=1)
                del expected_tokens, expected_mask
                torch.cuda.empty_cache()
            print(3.3, torch.cuda.mem_get_info(), torch.cuda.memory_reserved(), torch.cuda.memory_allocated())
            all_scores = torch.stack([x for x in all_scores], dim=1)
            # print('all_scores', all_scores.shape, all_scores)
            all_scores = all_scores.reshape(B, K, -1)
            # print('all_scores', all_scores.shape, all_scores)
            all_scores = torch.nanmean(all_scores, dim=-1)
            # print('all_scores', all_scores.shape, all_scores)
            del tokenized_queries, answers_input, answers_mask
            del expanded_answers_input, expanded_answers_mask, combined_input, combined_mask
            torch.cuda.empty_cache()
            return all_scores

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        opt = self.optimizers()
        print(1, torch.cuda.mem_get_info(), torch.cuda.memory_reserved(), torch.cuda.memory_allocated())
        # TODO: Make this actually work with the dataset constructed above
        # Pre-Process data (moved from above in order to batch tokenize for efficiency)
        questions = batch['questions']
        docs = batch['retrieved']
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
        self.llm = self.llm.to(device)
        self.llm.eval()
        print(1.01, torch.cuda.mem_get_info(), torch.cuda.memory_reserved(), torch.cuda.memory_allocated())

        print("1.1 size of inputs:", tokenized_questions['input_ids'].element_size()*tokenized_questions['input_ids'].nelement() + \
              tokenized_docs['input_ids'].element_size()*tokenized_docs['input_ids'].nelement() + \
              tokenized_answers['input_ids'].element_size()*tokenized_answers['input_ids'].nelement())


        # Run an LLM to get the NLLs (logits?)
        # llm_output = self.llm_pass(tokenized_questions, tokenized_docs, tokenized_answers)
        # llm_dist = torch.nn.functional.softmax(llm_output, dim=1)
        llm_output = self.llm_pass_2(questions, docs, tokenized_answers)
        llm_dist = torch.nn.functional.softmax(llm_output, dim=1)
        print(2, torch.cuda.mem_get_info(), torch.cuda.memory_reserved(), torch.cuda.memory_allocated())
        
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
        wandb.log({'train_loss': loss.item()})
        # opt.zero_grad()
        del tokenized_answers, rerank_dist, llm_output, llm_dist, lossfn
        torch.cuda.empty_cache()
        return loss

    def configure_optimizers(self):
        params = list(self.question_encoder.parameters()) + list(self.docs_encoder.parameters())
        return torch.optim.AdamW(params, lr=1e-4)

    def validation_step(self, batch, batch_idx):
        pass

    def inference(self, questions, docs, top_k):
        scores = self.forward(questions, docs)
        order = torch.argsort(scores, descending=True)
        scores, _ = torch.sort(scores, descending=True)
        return scores[:,:top_k], order[:,:top_k]
    
    def ensemble_predict(self, queries, docs_scores):
        B, K = docs_scores.shape

        combined_input = queries['input_ids']
        combined_mask = queries['attention_mask'].to(dtype=torch.bool)
        # print('combined_input', combined_input.shape)
        docs_scores = torch.unsqueeze(docs_scores, dim=2)
        all_preds = []
        for i in range(16):
            # print('combined_input', combined_input.shape)
            outputs = self.llm(input_ids=combined_input, attention_mask=combined_mask)                   # BK x (S+L+i) x V
            # print('outputs[logits]', outputs['logits'].shape)
            last_outputs = outputs['logits'][:, -1, :]           # BK x V
            # print('last_outputs', last_outputs.shape)
            # Aggregate across the documents of each question:
            last_outputs = torch.reshape(last_outputs, (B, K, -1)) # B x K x V
            expanded_docs_scores = docs_scores.expand(last_outputs.shape)
            # print('docs_scores', docs_scores.shape, docs_scores)
            last_outputs = last_outputs * expanded_docs_scores
            # print('last_outputs', last_outputs.shape)
            aggregate_outputs = torch.mean(last_outputs, dim=1)    # B x V
            # print('aggregate_outputs', aggregate_outputs.shape)
            pred_tokens = torch.argmax(aggregate_outputs, dim=1)   # B x 1
            pred_tokens = torch.unsqueeze(pred_tokens, dim=1)
            # print('pred_tokens', pred_tokens.shape)
            all_preds.append(pred_tokens)

            expanded_pred_tokens = self.expand_data(pred_tokens, B, K) # BK x 1
            pred_mask = torch.ones((B * K, 1)).to(device)

            combined_input = torch.cat([combined_input, expanded_pred_tokens], dim=1) #BK x (S + L + i)
            combined_mask = torch.cat([combined_mask, pred_mask], dim=1)
            del outputs, last_outputs, expanded_docs_scores, aggregate_outputs, expanded_pred_tokens, pred_mask
            torch.cuda.empty_cache()
        all_preds = torch.stack([x for x in all_preds], dim=1) # B x P
        all_preds = torch.reshape(all_preds, (B, -1))
        # print('all_preds', all_preds.shape, all_preds)
        decoded_preds = self.tokenizer.batch_decode(all_preds)
        decoded_preds = [pred.split('\n\n')[0] for pred in decoded_preds]
        print(decoded_preds)
        del all_preds, combined_input, combined_mask
        torch.cuda.empty_cache()
        return decoded_preds
    
    def ensemble_predict_2(self, questions, docs):
        if self.model_id.startswith('meta-llama'):
            query = [f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an assistant who gives short, succinct answers to questions. Please answer according to the following examples:<|eot_id|>

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
For this last question, use the following contexts to answer the question.
""" + \
'\n'.join(['Context: ' + d for d in doc]) +\
f"""
Question: {questions[i]}?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Answer:""" for i, doc in enumerate(docs)]
            
        elif self.model_id.startswith('microsoft'):
            query = [f"""<|system|>
You are an assistant who gives short, succinct answers to questions. Please answer in the following format:
Question: who was leander paes partner in the mixed doubles at the us open in 2008?
Answer: Cara Black

Question: who takes over after a president is impeached?
Answer: vice president

Question: who plays the dogs voice in downward dog?
Answer: Samm Hodges

Question: when did the name of persia change to iran?
Answer: 1935<|end|>
<|user|>
For this last question, use the following contexts to answer the question.
""" + \
'\n'.join(['Context: ' + d for d in doc]) +\
f"""
Question: {questions[i]}?<|end|>
<|assistant|>
Answer:""" for i, doc in enumerate(docs)]
            
        else:
            query = ["""Question: who was leander paes partner in the mixed doubles at the us open in 2008?
Answer: Cara Black

Question: who takes over after a president is impeached?
Answer: vice president

Question: who plays the dogs voice in downward dog?
Answer: Samm Hodges

Question: when did the name of persia change to iran?
Answer: 1935
""" + \
f'For the final question, answer using the given contexts:\n' + \
'\n'.join(['Context: ' + d for d in doc]) + \
f'\nQuestion:{questions[i]}?\nAnswer:' for i, doc in enumerate(docs)]
        # print(query)
        tokenized_query = self.tokenizer(query, padding=True, return_tensors='pt').to(device)
        print(tokenized_query['input_ids'].shape)
        outputs = self.llm.generate(**tokenized_query, max_new_tokens=16, tokenizer=self.tokenizer, stop_strings=['\n'])
        # print(outputs.shape)
        outputs = outputs[:,len(tokenized_query['input_ids'][0]):]
        # print(outputs.shape)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_outputs = [output.strip() for output in decoded_outputs]
        # print('decoded_outputs', decoded_outputs)
        del tokenized_query, outputs
        torch.cuda.empty_cache()
        # quit(0)
        return decoded_outputs

    def evaluate(self, valid_loader, top_k, rerank=True):
        temp = 1
        bleu = evaluate.load('bleu')
        bertscore = evaluate.load('bertscore')
        chrf = evaluate.load('chrf')
        em = evaluate.load('exact_match')
        predictions = []
        references = []

        self.question_encoder.to(device)
        self.docs_encoder.to(device)
        self.llm.to(device)
        self.question_encoder.eval()
        self.docs_encoder.eval()
        self.llm.eval()
        for batch in tqdm(valid_loader):
            questions = batch['questions']
            docs = batch['retrieved']
            answers = batch['answers']
            # print(np.asarray(questions).shape, np.asarray(docs).shape)
            # questions = ['When was the original Transformers movie released', 'What is the primary ingredient in Tabasco sauce']
            # docs = [['The first Transformers movie was released in 2007.', 'Tabasco sauce is composed mostly of vinegar, by volume.'], 
            #         ['Jurassic Park is a 1993 movie directed by Steven Spielberg', 'Potatoes are a starchy root vegetable.']]
            # answers = ['2007', 'vinegar']
            tokenized_questions = self.tokenizer(questions, padding=True, return_tensors='pt').to(device)
            tokenized_docs = self.tokenizer([x for retr in docs for x in retr], padding=True, return_tensors='pt').to(device)
            tokenized_docs['input_ids'] = torch.transpose(torch.reshape(tokenized_docs['input_ids'], (len(docs), len(docs[0]), -1)), 0, 1)
            tokenized_docs['attention_mask'] = torch.transpose(torch.reshape(tokenized_docs['attention_mask'], (len(docs), len(docs[0]), -1)), 0, 1)

            if rerank:
                rerank_scores, rerank_order = self.inference(tokenized_questions, tokenized_docs, top_k)
                rerank_scores = torch.nn.functional.softmax(rerank_scores / temp, dim=1)
                # print('rerank_scores', rerank_scores.shape, rerank_scores)
                # print(rerank_order)
                docs = np.asarray(docs).T
                # print(docs)
                new_docs = np.take_along_axis(np.asarray(docs), rerank_order.cpu().numpy(), axis=1)
                # print(np.asarray(new_docs))
                # queries = [['Please answer the following question using the following context. \n\nContext: ' + d + '\n\nQuestion: ' + questions[i] + '?\n\nVery short answer:' for d in doc] for i, doc in enumerate(new_docs)]
                # print(np.asarray(queries))
                # queries = [q for query in queries for q in query]
                # print(queries)
                # tokenized_queries = self.tokenizer(queries, padding='max_length', truncation=True, max_length=100, return_tensors='pt').to(device)
                # tokenized_queries = self.tokenizer(queries, padding=True, return_tensors='pt').to(device)
                
                # print('tokenized_new_docs', tokenized_new_docs['input_ids'].shape)
            else:
                # print(len(docs), len(docs[0]))
                new_docs = np.asarray(docs)[:top_k, :].T
                # print(new_docs)
                # print(len(new_docs), len(new_docs[0]))
            # preds = self.ensemble_predict(tokenized_queries, rerank_scores) # B x len(P)
            preds = self.ensemble_predict_2(questions, new_docs)

            # results = bleu.compute(predictions=preds, references=[[answer] for answer in answers])
            # results = bertscore.compute(predictions=preds, references=[[answer] for answer in answers], lang='en')
            # print(preds, [[answer] for answer in answers])
            # print(results)
            
            predictions += preds
            references += [[answer] for answer in answers]
            if rerank:
                del tokenized_questions, tokenized_docs, rerank_scores, rerank_order#, tokenized_queries
            else:
                del tokenized_questions, tokenized_docs
            torch.cuda.empty_cache()

            # new_queries = ['Please answer the following question using as few words as possible.\n\nQuestion: ' + question + '?\n\nVery short answer: ' for question in questions]
            # print('new_queries', new_queries)
            # # new_preds = self.ensemble_predict(tokenized_new_queries, torch.ones((len(questions), 1)).to(device))
            
            # tokenized_new_queries = self.tokenizer(new_queries, padding=True, return_tensors='pt').to(device)
            # tokenized_new_query_inputs, tokenized_new_query_masks = tokenized_new_queries['input_ids'], tokenized_new_queries['attention_mask']
            # new_preds = self.llm.generate(input_ids=tokenized_new_query_inputs, attention_mask=tokenized_new_query_masks, max_new_tokens=16,
            #                               tokenizer=self.tokenizer, stop_strings=['\n\n'])
            # new_preds = self.tokenizer.batch_decode(new_preds)
            # print('new_preds', new_preds)

            print(preds, answers)
            # print(bleu.compute(predictions=predictions, references=references))
            # quit(0)
        bleu_results = bleu.compute(predictions=predictions, references=references)
        bertscore_results = bertscore.compute(predictions=predictions, references=references, lang='en')
        chrf_results = chrf.compute(predictions=predictions, references=references, word_order=1)
        em_results = em.compute(predictions=predictions, references=list(np.asarray(references).flatten()), ignore_case=True, ignore_punctuation=True)
        print(bleu_results)
        print(bertscore_results)
        print(chrf_results)
        print(em_results)