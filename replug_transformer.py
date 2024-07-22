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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.vocab_size = self.tokenizer.vocab_size
        self.question_encoder = Encoder(self.vocab_size)
        self.docs_encoder = Encoder(self.vocab_size)
        self.llm = AutoModelForCausalLM.from_pretrained(self.model_id)  # LLM to get NLLs for reference distribution in KL div 
        for param in self.llm.parameters():
            param.requires_grad = False
    
    def forward(self, questions, docs):
        questions_input, questions_mask = questions['input_ids'], torch.logical_not(questions['attention_mask'].to(dtype=torch.bool))
        docs_input, docs_mask = docs['input_ids'], torch.logical_not(docs['attention_mask'].to(dtype=torch.bool))
        # Encode questions and documents
        # print('questions', questions)
        # print(len(docs), questions.shape, docs[0].shape)
        # print('docs', docs.shape, docs)
        n_examples = questions_input.shape[0]
        q_emb = self.question_encoder(questions_input, questions_mask)
        # print('q_emb', q_emb.shape, q_emb)
        # Squeeze and unsqueeze to pass batch of retrievals into encoder at once?
        # Turn B x K x L to (BK) x L
        B, K, L = docs_input.shape
        # print(B, K, L)
        squeeze_docs_input = torch.reshape(docs_input, (B * K, L))
        squeeze_docs_mask = torch.reshape(docs_mask, (B*K, -1))
        # print(input_docs.shape)
        d_embs = self.docs_encoder(squeeze_docs_input, squeeze_docs_mask)
        d_embs = torch.reshape(d_embs, (B, K, -1))
        # print(d_embs.shape)

        d_scores = torch.einsum('bij,bjk->bik', d_embs, torch.unsqueeze(q_emb, -1))
        d_scores = d_scores.squeeze()

        del questions_input, questions_mask, docs_input, docs_mask, squeeze_docs_input, squeeze_docs_mask, q_emb, d_embs
        torch.cuda.empty_cache()
        # print('d_scores', d_scores.shape, d_scores)

        return d_scores
    
    def expand_data(self, data, batch_size, num_docs):                   # Data: B x S
        expanded_data = torch.unsqueeze(data, 1)                         # B x 1 x S
        expanded_data = expanded_data.expand(-1, num_docs, -1)           # B x K x S
        return torch.reshape(expanded_data, (batch_size * num_docs, -1)) # BK x S

    def new_llm_pass(self, questions, docs, answers):
        with torch.no_grad():
            # Again, reshape to put into LLM as B x ? shape
            questions_input, questions_mask = questions['input_ids'], torch.logical_not(questions['attention_mask'].to(dtype=torch.bool))
            docs_input, docs_mask = docs['input_ids'], torch.logical_not(docs['attention_mask'].to(dtype=torch.bool))
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
            # print("expanded_answers_mask", expanded_answers_mask)
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
                # print("expected_tokens_mask", expected_tokens_mask)
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

    def training_step(self, batch, batch_idx):
        # TODO: Make this actually work with the dataset constructed above
        # Pre-Process data (moved from above in order to batch tokenize for efficiency)
        questions = batch['questions']
        docs = batch['retrieved']
        answers = batch['answers']
        tokenized_questions = self.tokenizer(questions, padding=True, return_tensors='pt').to(device)
        tokenized_docs = self.tokenizer([x for retr in docs for x in retr], padding='max_length', truncation=True, max_length=100, return_tensors='pt').to(device)
        tokenized_docs['input_ids'] = torch.transpose(torch.reshape(tokenized_docs['input_ids'], (len(docs), len(docs[0]), -1)), 0, 1)
        tokenized_docs['attention_mask'] = torch.transpose(torch.reshape(tokenized_docs['attention_mask'], (len(docs), len(docs[0]), -1)), 0, 1)
        tokenized_answers = self.tokenizer(answers, padding=True, return_tensors='pt').to(device)

        self.question_encoder = self.question_encoder.to(device)
        self.docs_encoder = self.docs_encoder.to(device)
        self.llm = self.llm.to(device)
        
        # Normalize the retrieval scores from the forward pass
        # TODO: Apply the masks in the scoring process - currently no masks but still converges on single batch

        reranker_output = self(tokenized_questions, tokenized_docs)  # output is retrieval scores?
        rerank_dist = torch.nn.functional.log_softmax(reranker_output, dim=1)
        
        # Run an LLM to get the NLLs - NLLs or is it just the logits??
        llm_output = self.new_llm_pass(tokenized_questions, tokenized_docs, tokenized_answers)
        llm_dist = torch.nn.functional.softmax(llm_output, dim=1)

        # print('rerank', rerank_dist.shape, rerank_dist)
        # print(rerank_dist)
        # print('llm', llm_dist.shape, llm_dist)
        # print(llm_dist)

        # Compute loss = kldiv(scores, nlls)
        lossfn = torch.nn.KLDivLoss(reduction='batchmean')
        loss = lossfn(rerank_dist, llm_dist) # see docs for notation https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
        # print(loss)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(questions))
        wandb.log({'train_loss': loss.item()})
        del tokenized_questions, tokenized_docs, tokenized_answers, reranker_output, rerank_dist, llm_output, llm_dist, lossfn
        return loss

    def configure_optimizers(self):
        params = list(self.question_encoder.parameters()) + list(self.docs_encoder.parameters())
        return torch.optim.AdamW(params, lr=1e-4)
    
    def inference(self, questions, docs, top_k):
        scores = self.forward(questions, docs)
        order = torch.argsort(scores, descending=True)
        scores, _ = torch.sort(scores, descending=True)
        return scores[:,:top_k], order[:,:top_k]
    
    def ensemble_predict(self, questions, docs, docs_scores):
        questions_input, questions_mask = questions['input_ids'], torch.logical_not(questions['attention_mask'].to(dtype=torch.bool))
        docs_input, docs_mask = docs['input_ids'], torch.logical_not(docs['attention_mask'].to(dtype=torch.bool))
        B, K, L = docs_input.shape
        docs_input = torch.reshape(docs_input, (B*K, L))                   # BK x L
        docs_mask = torch.reshape(docs_mask, (B*K, L))         # BK x L
        # _, answer_length = answers_input.shape
        # print('questions_input', questions_input.shape)
        expanded_questions_input = self.expand_data(questions_input, B, K)
        expanded_questions_mask = self.expand_data(questions_mask, B, K)
        # print('expanded_questions', expanded_questions.shape, expanded_questions)
        # TODO: reconsider how the inputs are combined, right now they are just concatenated with the padding... Retokenize?
        combined_input = torch.cat([docs_input, expanded_questions_input], dim=1)      # BK x (S + L)
        combined_mask = torch.cat([docs_mask, expanded_questions_mask], dim=1) # BK x (S + L)
        # print('combined_input', combined_input.shape)
        # expanded_answers_input = self.expand_data(answers_input, B, K)
        # expanded_answers_mask = self.expand_data(answers_mask, B, K)
        # print("expanded_answers_mask", expanded_answers_mask)
        docs_scores = torch.unsqueeze(docs_scores, dim=2)
        all_preds = []
        for i in range(8):
            # print('combined_input', combined_input.shape)
            # expected_tokens = expanded_answers_input[:, i]             # BK x 1
            # expected_mask = expanded_answers_mask[:, i]   # BK x 1
            # print('expected_tokens', expected_tokens.shape, expected_tokens)
            outputs = self.llm(combined_input, combined_mask)                   # BK x (S+L+i) x V
            # print('outputs[logits]', outputs['logits'].shape)
            last_outputs = outputs['logits'][:, -1, :]           # BK x V
            # print('last_outputs', last_outputs.shape)
            # Aggregate across the documents of each question:
            last_outputs = torch.reshape(last_outputs, (B, K, -1)) # B x K x V
            # print("docs_scores", docs_scores.shape, docs_scores)
            
            # print("docs_scores", docs_scores.shape, docs_scores)
            expanded_docs_scores = docs_scores.expand(last_outputs.shape)
            # print("docs_scores", docs_scores.shape, docs_scores)
            last_outputs = last_outputs * expanded_docs_scores
            # print('last_outputs', last_outputs.shape)
            aggregate_outputs = torch.mean(last_outputs, dim=1)    # B x V
            # print('aggregate_outputs', aggregate_outputs.shape)
            # scores = last_outputs[torch.arange(B*K), expected_tokens]
            pred_tokens = torch.argmax(aggregate_outputs, dim=1)   # B x 1
            pred_tokens = torch.unsqueeze(pred_tokens, dim=1)
            # print('pred_tokens', pred_tokens.shape)
            all_preds.append(pred_tokens)

            expanded_pred_tokens = self.expand_data(pred_tokens, B, K) # BK x 1
            pred_mask = torch.ones((B * K, 1)).to(device)
            # print('scores', scores.shape, scores)
            # print("expected_tokens_mask", expected_tokens_mask)
            # scores = torch.where(expected_mask == 1, scores, torch.nan)
            # print('scores', scores.shape, scores)
            # all_scores.append(scores)

            combined_input = torch.cat([combined_input, expanded_pred_tokens], dim=1) #BK x (S + L + i)
            combined_mask = torch.cat([combined_mask, pred_mask], dim=1)
            # del expected_tokens, expected_mask, outputs, last_outputs
            # torch.cuda.empty_cache()
        all_preds = torch.stack([x for x in all_preds], dim=1) # B x P
        all_preds = torch.reshape(all_preds, (B, -1))
        # print('all_preds', all_preds.shape, all_preds)
        decoded_preds = self.tokenizer.batch_decode(all_preds)
        print(decoded_preds)
        # print('all_scores', all_scores.shape, all_scores)
        # all_scores = all_scores.reshape(B, K, -1)
        # print('all_scores', all_scores.shape, all_scores)
        # all_scores = torch.nanmean(all_scores, dim=-1)
        return decoded_preds

    def eval(self, valid_loader, top_k, rerank=True):
        bleu = evaluate.load('bleu')
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
            
            tokenized_questions = self.tokenizer(questions, padding=True, return_tensors='pt').to(device)
            tokenized_docs = self.tokenizer([x for retr in docs for x in retr], padding='max_length', truncation=True, max_length=100, return_tensors='pt').to(device)
            tokenized_docs['input_ids'] = torch.transpose(torch.reshape(tokenized_docs['input_ids'], (len(docs), len(docs[0]), -1)), 0, 1)
            tokenized_docs['attention_mask'] = torch.transpose(torch.reshape(tokenized_docs['attention_mask'], (len(docs), len(docs[0]), -1)), 0, 1)
            # tokenized_answers = self.tokenizer(answers, padding=True, return_tensors='pt').to(device)

            rerank_scores, rerank_order = self.inference(tokenized_questions, tokenized_docs, top_k)
            rerank_scores = torch.nn.functional.softmax(rerank_scores, dim=1)
            # print(rerank_scores)
            # print(rerank_order)
            new_docs = np.take(docs, rerank_order.cpu())
            queries = [['Context: ' + d + " Question: " + questions[i] + "? Answer: " for i, d in enumerate(doc)] for doc in new_docs]
            print(queries)
            queries = [q for query in queries for q in query]
            tokenized_new_docs = self.tokenizer([x for retr in new_docs for x in retr], padding='max_length', truncation=True, max_length=100, return_tensors='pt').to(device)
            tokenized_new_docs['input_ids'] = torch.reshape(tokenized_new_docs['input_ids'], (len(new_docs), len(new_docs[0]), -1))
            tokenized_new_docs['attention_mask'] = torch.reshape(tokenized_new_docs['attention_mask'], (len(new_docs), len(new_docs[0]), -1))
            
            # print("tokenized_new_docs", tokenized_new_docs['input_ids'].shape)
            preds = self.ensemble_predict(tokenized_questions, tokenized_new_docs, rerank_scores) # B x len(P)

            # results = bleu.compute(predictions=preds, references=[[answer] for answer in answers])
            # print(results)
            answers = batch['answers']
            predictions += preds
            references += [[answer] for answer in answers]
        results = bleu.compute(predictions=predictions, references=references)
        print(results)