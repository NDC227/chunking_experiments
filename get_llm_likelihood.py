import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
from datasets import load_dataset, Dataset
import argparse
from multiprocess import set_start_method
import os

# python get_llm_likelihood.py --llm_name facebook/opt-125m --dataset rechunked_nq --output_name rechunked_and_scored_nq --tiny --max_seq_length 600 --batch_size 64
# python get_llm_likelihood.py --llm_name facebook/opt-125m --dataset rechunked_nq --output_name rechunked_and_scored_nq --split test --tiny --max_seq_length 600 --batch_size 64
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
argp.add_argument('--debug', action='store_true')
args = argp.parse_args()

os.environ['HF_HOME'] = '/nlp/scr/ayc227/.cache/'
os.environ['HF_TOKEN'] = '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'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.set_float32_matmul_precision('medium')

num_proc = args.num_proc
batch_size = args.batch_size
model_id = args.llm_name
output_name = args.output_name
if args.tiny:
    output_name = 'toy_' + output_name

def print_debug(*prints):
    if args.debug:
        print(prints)

def expand_data(data, batch_size, num_docs):                         # Data: B x S
    expanded_data = torch.unsqueeze(data, 1)                         # B x 1 x S
    expanded_data = expanded_data.expand(-1, num_docs, -1)           # B x K x S
    return torch.reshape(expanded_data, (batch_size * num_docs, -1)) # BK x S

def batch_llm_pass(questions, docs, answers, device):
    with torch.no_grad():
        if model_id.startswith('meta-llama'):
            queries = [f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
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
Answer:''' for i in range(len(docs))]
            
        elif model_id.startswith('microsoft'):
            queries = [f'''<|system|>
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
Answer:''' for i in range(len(docs))]
            
        else:
            queries = [f'''You are an assistant who gives short, succinct answers to questions. Please answer the following questions using the contexts given below:
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
Answer:''' for i in range(len(docs))]
        
        tokenizer.padding_side = 'left'
        tokenized_queries = tokenizer(queries, padding=True, truncation=True, max_length=args.max_seq_length, return_tensors='pt').to(device)
        answers_input, answers_mask = answers['input_ids'], answers['attention_mask'].to(dtype=torch.bool) # B x A
        # print_debug('answers_input', answers_input.shape, answers_input)
        print_debug(3.1, torch.cuda.mem_get_info(device), torch.cuda.memory_reserved(device), torch.cuda.memory_allocated(device))

        _, answer_length = answers_input.shape
        combined_input = tokenized_queries['input_ids']      # BK x (S + L)
        combined_mask = tokenized_queries['attention_mask'] # BK x (S + L)
        # print_debug('combined_input', combined_input.shape)
        all_scores = []
        for i in range(answer_length):
            print_debug(3.2, torch.cuda.mem_get_info(device), i, torch.cuda.memory_reserved(device), torch.cuda.memory_allocated(device))
            expected_tokens = answers_input[:, i]             # B x 1
            expected_mask = answers_mask[:, i]   # B x 1
            outputs = llm(input_ids=combined_input, attention_mask=combined_mask)                   # BK x (S+L+i) x V
            # print_debug('expected_tokens', expected_tokens.shape)
            # print_debug('outputs[logits]', outputs['logits'].shape)
            # print_debug('combined_input size', combined_input.element_size()*combined_input.nelement())
            # print_debug('outputs size', outputs['logits'].element_size()*outputs['logits'].nelement())
            print_debug('3.2.1', torch.cuda.mem_get_info(device), i, torch.cuda.memory_reserved(device), torch.cuda.memory_allocated(device))
            last_outputs = outputs['logits'][:, -1, :]           # BK x V
            # print_debug('last_outputs', last_outputs.shape)
            scores = (last_outputs[torch.arange(expected_tokens.shape[0]), expected_tokens]) # BK x 1
            del outputs, last_outputs
            torch.cuda.empty_cache()
            # print_debug('expected_tokens_mask', expected_tokens_mask)
            scores = torch.where(expected_mask == 1, scores, torch.nan)
            # print_debug('scores', scores.shape, scores)
            all_scores.append(scores)

            combined_input = torch.cat([combined_input, torch.unsqueeze(expected_tokens, 1)], dim=1) #BK x (S + L + i)
            combined_mask = torch.cat([combined_mask, torch.unsqueeze(expected_mask, 1)], dim=1)
            del expected_tokens, expected_mask
            torch.cuda.empty_cache()
        print_debug(3.3, torch.cuda.mem_get_info(device), torch.cuda.memory_reserved(device), torch.cuda.memory_allocated(device))
        all_scores = torch.stack([x for x in all_scores], dim=1)
        all_scores = torch.nanmean(all_scores, dim=-1)
        print_debug('all_scores', all_scores.shape, all_scores)
        del tokenized_queries, answers_input, answers_mask
        del combined_input, combined_mask
        torch.cuda.empty_cache()
        return all_scores

def batch_score_chunks(batch, rank):
    device = f'cuda:{(rank or 0) % torch.cuda.device_count()}'
    llm.to(device)
    questions = batch['questions']
    docs = batch['new_chunks']
    answers = batch['answers']
    tokenizer.padding_side = 'right'
    tokenized_answers = tokenizer(answers, padding=True, return_tensors='pt').to(device)
    llm_output = batch_llm_pass(questions, docs, tokenized_answers, device)
    batch['llm_scores'] = llm_output
    return batch

if args.tiny:
    train_chunks = load_dataset(f'{args.user}/{args.dataset}', split=args.split, streaming=True, cache_dir='/nlp/scr/ayc227/.cache/huggingface/datasets').take(10)
    train_chunks = Dataset.from_generator(lambda: (yield from train_chunks), features=train_chunks.features)
else:
    train_chunks = load_dataset(f'{args.user}/{args.dataset}', split=args.split, num_proc=torch.cuda.device_count(), cache_dir='/nlp/scr/ayc227/.cache/huggingface/datasets')

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir='/nlp/scr/ayc227/.cache/huggingface/models')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

if model_id.startswith('facebook'):
    llm = AutoModelForCausalLM.from_pretrained(model_id, cache_dir='/nlp/scr/ayc227/.cache/huggingface/models')
else:
    llm = AutoModelForCausalLM.from_pretrained(model_id, cache_dir='/nlp/scr/ayc227/.cache/huggingface/models', 
                                               torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2')
llm.resize_token_embeddings(len(tokenizer))
llm.eval()

def flatten_chunks(batch, rank):
    idxs = [np.random.choice(len(batch['new_chunks'][i]), size=(args.downsample), replace=False) for i in range(len(batch['new_chunks']))]
    new_batch = {}
    new_batch['questions'] = [questions for i, questions in enumerate(batch['questions']) for _ in range(args.downsample)]
    new_batch['answers'] = [answers for i, answers in enumerate(batch['answers']) for _ in range(args.downsample)]
    new_batch['new_chunks'] = [chunks[idx] for i, chunks in enumerate(batch['new_chunks']) for idx in idxs[i]]
    new_batch['chunker_ids'] = [chunks[idx] for i, chunks in enumerate(batch['chunker_ids']) for idx in idxs[i]]
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

    flat_train_chunks = train_chunks.map(flatten_chunks, batched=True, batch_size=16, with_rank=True, num_proc=torch.cuda.device_count())
    print_debug(flat_train_chunks)
    print('chunks flattened')
    
    flat_train_chunks = flat_train_chunks.map(batch_score_chunks, batched=True, batch_size=args.batch_size, with_rank=True, num_proc=torch.cuda.device_count())
    print_debug(flat_train_chunks)
    print('llm scores calculated')

    train_chunks = train_chunks.map(merge_chunks, num_proc=torch.cuda.device_count())
    print_debug(train_chunks)
    print_debug('length of first new_chunks', len(train_chunks[0]['new_chunks']))
    print('llm scores remapped to original dataset')

    train_chunks.push_to_hub(f'{args.user}/{output_name}', private=True)
    print('done uploading to hub')
