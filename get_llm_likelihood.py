import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
from datasets import load_dataset, Dataset
import argparse
from torch.utils.data import DataLoader
from multiprocess import set_start_method

import os
os.environ['HF_HOME'] = '/nlp/scr/ayc227/.cache/'
os.environ['HF_TOKEN'] = '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'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision('medium')
# device = 'cuda' if torch.cuda.is_available else 'cpu'

# python get_llm_likelihood.py --llm_name facebook/opt-125m --dataset rechunked_nq --output_name rechunked_and_scored_nq --tiny --max_seq_length 600 --batch_size 64
# python get_llm_likelihood.py --user ContextualAI --llm_name meta-llama/Meta-Llama-3.1-8B-Instruct --dataset rechunked_nq --output_name rechunked_and_scored_nq --max_seq_length 600 --batch_size 32 
argp = argparse.ArgumentParser()
argp.add_argument('--llm_name', default='facebook/opt-125m')
argp.add_argument('--dataset', default='rechunked_nq')
argp.add_argument('--user', default='ndc227')
argp.add_argument('--output_name', default='rechunked_and_scored_nq')
argp.add_argument('--num_proc', default=1, type=int)
argp.add_argument('--batch_size', default=2, type=int)
argp.add_argument('--max_seq_length', default=400, type=int)
argp.add_argument('--split', default='train')
argp.add_argument('--tiny', action='store_true')
argp.add_argument('--downsample', default=100, type=int)
args = argp.parse_args()

num_proc = args.num_proc
batch_size = args.batch_size
model_id = args.llm_name

def expand_data(data, batch_size, num_docs):                   # Data: B x S
    expanded_data = torch.unsqueeze(data, 1)                         # B x 1 x S
    expanded_data = expanded_data.expand(-1, num_docs, -1)           # B x K x S
    return torch.reshape(expanded_data, (batch_size * num_docs, -1)) # BK x S

def llm_pass(questions, docs, answers, device):
    with torch.no_grad():
        docs = np.asarray(docs)
        B, K = len(docs), len(docs[0])
        # print(B, K)
        if model_id.startswith('meta-llama'):
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
            
        elif model_id.startswith('microsoft'):
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
        tokenized_queries = tokenizer(queries, padding=True, return_tensors='pt').to(device)
        answers_input, answers_mask = answers['input_ids'], answers['attention_mask'].to(dtype=torch.bool) # B x A
        # print('answers_input', answers_input.shape, answers_input)
        print(3.1, torch.cuda.mem_get_info(device), torch.cuda.memory_reserved(device), torch.cuda.memory_allocated(device))

        _, answer_length = answers_input.shape
        combined_input = tokenized_queries['input_ids']      # BK x (S + L)
        combined_mask = tokenized_queries['attention_mask'] # BK x (S + L)
        # print('combined_input', combined_input.shape)
        expanded_answers_input = expand_data(answers_input, B, K)
        # expanded_answers_mask = expand_data(answers_mask, B, K)
        # print('expanded_answers_input', expanded_answers_input.shape, expanded_answers_input)
        all_scores = []
        for i in range(answer_length):
            print(3.2, torch.cuda.mem_get_info(device), i, torch.cuda.memory_reserved(device), torch.cuda.memory_allocated(device))
            # print('combined_input', combined_input.shape)
            expected_tokens = expanded_answers_input[:, i]        # BK x 1
            # expected_mask = expanded_answers_mask[:, i]           # BK x 1
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
            split_factor = 100
            j = 0
            while B*K//split_factor*j < B*K:
                start = B*K//split_factor*j
                end = min(B*K,B*K//split_factor*(j+1))
                # print(start)
                # print(combined_input.shape, combined_input)
                # print(B*K//10*j, B*K//10*(j+1))
                # print(combined_input[B*K//10*j:B*K//10*(j+1),:].shape, combined_input[B*K//10*j:B*K//10*(j+1),:])
                # print(combined_mask[B*K//10*j:B*K//10*(j+1),:].shape, combined_mask[B*K//10*j:B*K//10*(j+1),:])
                outputs = llm(input_ids=combined_input[start:end,:], 
                              attention_mask=combined_mask[start:end,:])                   # BK//split_factor x (S+L+i) x V
                # print('outputs[logits]', outputs['logits'].shape)
                last_outputs = outputs['logits'][:, -1, :]           # BK//split_factor x V
                # print('last_outputs', last_outputs.shape)
                # print('expected_tokens[start:end]', expected_tokens[start:end].shape)
                score = last_outputs[torch.arange(end-start),expected_tokens[start:end]] # BK//split_factor x 1
                # print('score', score.shape) 
                scores.append(score) # BK x 1
                del outputs, last_outputs, score
                torch.cuda.empty_cache()
                j += 1
            scores = torch.cat(scores, dim=0) # BK x 1
            # print('scores', scores.shape)
            # print('expected_mask', expected_mask.shape)
            # scores = torch.where(expected_mask == 1, scores, torch.nan)
            # print('scores', scores.shape, scores)
            all_scores.append(scores)

            combined_input = torch.cat([combined_input, expected_tokens.unsqueeze(1)], dim=1) #BK x (S + L + i)
            combined_mask = torch.cat([combined_mask, torch.ones(expected_tokens.unsqueeze(1).shape).to(device)], dim=1)
            del expected_tokens#, expected_mask
            torch.cuda.empty_cache()
        print(3.3, torch.cuda.mem_get_info(device), torch.cuda.memory_reserved(device), torch.cuda.memory_allocated(device))
        all_scores = torch.stack([x for x in all_scores], dim=1)
        # print('all_scores', all_scores.shape, all_scores)
        all_scores = all_scores.reshape(B, K, -1)
        # print('all_scores', all_scores.shape, all_scores)
        all_scores = torch.nanmean(all_scores, dim=-1) # B x K
        print('all_scores', all_scores.shape, all_scores)
        del tokenized_queries, answers_input, answers_mask
        del combined_input, combined_mask, expanded_answers_input#, expanded_answers_mask
        torch.cuda.empty_cache()
        return all_scores
    
def llm_batch_pass(questions, docs, answers, device):
    with torch.no_grad():
        # docs = np.asarray(docs)
        B = len(docs)
        # print(B)
        if model_id.startswith('meta-llama'):
            queries = [f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an assistant who gives short, succinct answers to questions. Please answer the following questions using the context given below: 
Context: {docs[i]}<|eot_id|>

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
Answer:""" for i in range(len(docs))]
            
        elif model_id.startswith('microsoft'):
            queries = [f"""<|system|>
You are an assistant who gives short, succinct answers to questions. Please answer the following questions using the contexts given below: 
Context: {docs[i]}<|end|>
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
Answer:""" for i in range(len(docs))]
            
        else:
            queries = [f"""You are an assistant who gives short, succinct answers to questions. Please answer the following questions using the contexts given below:
Context: {docs[i]}
                       
Question: who was leander paes partner in the mixed doubles at the us open in 2008?
Answer: Cara Black

Question: who takes over after a president is impeached?
Answer: vice president

Question: who plays the dogs voice in downward dog?
Answer: Samm Hodges

Question: when did the name of persia change to iran?
Answer: 1935

Question:{questions[i]}?
Answer:""" for i in range(len(docs))]
        
        # queries = [['Please answer the following question using the following context. The answer should not restate the question.\n\nContext: ' + d + '\n\nQuestion: ' + questions[i] + '?\n\nVery short answer:' for d in doc] for i, doc in enumerate(docs)]
        # queries = [q for query in queries for q in query]
        # print(queries[:2])
        # quit(0)
        tokenizer.padding_side = 'left'
        tokenized_queries = tokenizer(queries, padding=True, truncation=True, max_length=args.max_seq_length, return_tensors='pt').to(device)
        answers_input, answers_mask = answers['input_ids'], answers['attention_mask'].to(dtype=torch.bool) # B x A
        # print('answers_input', answers_input.shape, answers_input)
        print(3.1, torch.cuda.mem_get_info(device), torch.cuda.memory_reserved(device), torch.cuda.memory_allocated(device))

        _, answer_length = answers_input.shape
        combined_input = tokenized_queries['input_ids']      # BK x (S + L)
        combined_mask = tokenized_queries['attention_mask'] # BK x (S + L)
        print('combined_input', combined_input.shape)
        # expanded_answers_input = expand_data(answers_input, B, K)
        # expanded_answers_mask = expand_data(answers_mask, B, K)
        # print('expanded_answers_mask', expanded_answers_mask)
        all_scores = []
        for i in range(answer_length):
            # print(3.2, torch.cuda.mem_get_info(device), i, torch.cuda.memory_reserved(device), torch.cuda.memory_allocated(device))
            # print('combined_input', combined_input.shape)
            expected_tokens = answers_input[:, i]             # B x 1
            expected_mask = answers_mask[:, i]   # B x 1
            # print('expected_tokens', expected_tokens.shape)

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
            outputs = llm(input_ids=combined_input, 
                          attention_mask=combined_mask)                   # BK x (S+L+i) x V
            # print('outputs[logits]', outputs['logits'].shape)
            # print('outputs[logits]', outputs['logits'].shape)
            # print('combined_input size', combined_input.element_size()*combined_input.nelement())
            # print('outputs size', outputs['logits'].element_size()*outputs['logits'].nelement())
            # print('3.2.1', torch.cuda.mem_get_info(device), i, torch.cuda.memory_reserved(device), torch.cuda.memory_allocated(device))
            # quit(0)
            last_outputs = outputs['logits'][:, -1, :]           # BK x V
            # print('last_outputs', last_outputs.shape)
            scores = (last_outputs[torch.arange(expected_tokens.shape[0]), expected_tokens]) # BK x 1
            del outputs, last_outputs
            torch.cuda.empty_cache()
            # print('scores', scores.shape, scores)
            # print('expected_tokens_mask', expected_tokens_mask)
            scores = torch.where(expected_mask == 1, scores, torch.nan)
            # print('scores', scores.shape, scores)
            all_scores.append(scores)

            combined_input = torch.cat([combined_input, torch.unsqueeze(expected_tokens, 1)], dim=1) #BK x (S + L + i)
            combined_mask = torch.cat([combined_mask, torch.unsqueeze(expected_mask, 1)], dim=1)
            del expected_tokens, expected_mask
            torch.cuda.empty_cache()
        print(3.3, torch.cuda.mem_get_info(device), torch.cuda.memory_reserved(device), torch.cuda.memory_allocated(device))
        all_scores = torch.stack([x for x in all_scores], dim=1)
        # print('all_scores', all_scores.shape, all_scores)
        # all_scores = all_scores.reshape(B, K, -1)
        # print('all_scores', all_scores.shape, all_scores)
        all_scores = torch.nanmean(all_scores, dim=-1)
        print('all_scores', all_scores.shape, all_scores)
        del tokenized_queries, answers_input, answers_mask
        del combined_input, combined_mask
        torch.cuda.empty_cache()
        return all_scores

def score_chunks(ex, rank):
    device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
    print(device)
    llm.to(device)
    questions = [ex['questions']]
    # print(questions)
    docs = [ex['new_chunks']]
    answers = [ex['answers']]
    tokenized_answers = tokenizer(answers, padding=True, return_tensors='pt').to(device)
    llm_output = llm_pass(questions, docs, tokenized_answers, device)
    llm_dist = torch.nn.functional.softmax(llm_output, dim=1)
    # print(len(docs))
    ex['llm_scores'] = llm_dist
    return ex

def score_chunks_batch(batch, rank):
    device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
    llm.to(device)
    questions = batch['questions']
    # print(questions)
    docs = batch['new_chunks']
    answers = batch['answers']
    tokenizer.padding_side = 'right'
    tokenized_answers = tokenizer(answers, padding=True, return_tensors='pt').to(device)
    llm_output = llm_batch_pass(questions, docs, tokenized_answers, device)
    # llm_dist = torch.nn.functional.softmax(llm_output, dim=1)
    # print(len(docs))
    batch['llm_scores'] = llm_output
    return batch

if args.tiny:
    train_chunks = load_dataset(f'{args.user}/{args.dataset}', split=args.split, streaming=True, cache_dir='/nlp/scr/ayc227/.cache/huggingface/datasets').take(3610)
    train_chunks = Dataset.from_generator(lambda: (yield from train_chunks), features=train_chunks.features)
else:
    train_chunks = load_dataset(f'{args.user}/{args.dataset}', split=args.split, num_proc=torch.cuda.device_count(), cache_dir='/nlp/scr/ayc227/.cache/huggingface/datasets')

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir='/nlp/scr/ayc227/.cache/huggingface/models')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

if model_id.startswith('facebook'):
    llm = AutoModelForCausalLM.from_pretrained(model_id, cache_dir='/nlp/scr/ayc227/.cache/huggingface/models')
else:
    llm = AutoModelForCausalLM.from_pretrained(model_id, cache_dir='/nlp/scr/ayc227/.cache/huggingface/models', 
                                               torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
llm.resize_token_embeddings(len(tokenizer))
llm.eval()

def flatten_chunks(batch, rank):
    # print(batch['new_chunks'])
    idxs = [np.random.choice(len(batch['new_chunks'][i]), size=(args.downsample), replace=False) for i in range(len(batch['new_chunks']))]
    # print(idxs)
    new_batch = {}
    new_batch['questions'] = [questions for i, questions in enumerate(batch['questions']) for _ in range(args.downsample)]
    new_batch['answers'] = [answers for i, answers in enumerate(batch['answers']) for _ in range(args.downsample)]
    
    new_batch['new_chunks'] = [chunks[idx] for i, chunks in enumerate(batch['new_chunks']) for idx in idxs[i]]
    new_batch['chunker_ids'] = [chunks[idx] for i, chunks in enumerate(batch['chunker_ids']) for idx in idxs[i]]
    # print(len(new_batch['new_chunks']))
    # quit(0)
    return new_batch

def merge_chunks(ex):
    question = ex['questions']
    filtered_flat_chunks = flat_train_chunks.filter(lambda x: x['questions']==question)
    ex['new_chunks'] = filtered_flat_chunks['new_chunks']
    ex['chunker_ids'] = filtered_flat_chunks['chunker_ids']
    ex['llm_scores'] = filtered_flat_chunks['llm_scores']
    return ex


if __name__ == '__main__':
    set_start_method('spawn')

    train_chunks = train_chunks.remove_columns('retrieved')
    # train_chunks = train_chunks.map(score_chunks, with_rank=True, num_proc=torch.cuda.device_count())

    flat_train_chunks = train_chunks.map(flatten_chunks, batched=True, batch_size=16, with_rank=True, num_proc=torch.cuda.device_count())
    print(flat_train_chunks)
    
    flat_train_chunks = flat_train_chunks.map(score_chunks_batch, batched=True, batch_size=args.batch_size, with_rank=True, num_proc=torch.cuda.device_count())
    print(flat_train_chunks)

    train_chunks = train_chunks.map(merge_chunks, num_proc=torch.cuda.device_count())
    print(train_chunks)
    print(len(train_chunks[0]['new_chunks']))

    train_chunks.push_to_hub(f'{args.user}/{args.output_name}', private=True)
    print('done uploading to hub')
