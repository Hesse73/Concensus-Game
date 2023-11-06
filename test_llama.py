from transformers import AutoModelForCausalLM, AutoTokenizer, top_k_top_p_filtering
import torch
import numpy as np


def gen_task(Q, A, A_id, correct):
    instruct = '\nAnswer:' if correct else '\nIncorrect answer:'
    return f'{Q}\n' + '\n'.join(['{}. {}'.format(x, y) for x, y in zip(A_id, A)]) + instruct


def dis_task(Q, a, incor_text='False', cor_text='True'):
    return f'"The answer to "{Q}" is {a}."\nThe claim above is ({cor_text} or {incor_text}):\nAnswer:'

# def dis_task(Q, a, cor_text='True', incor_text='False'):
#     return f'"The answer to "{Q}" is {a}."\nThe claim above is?\nA. correct\nB. incorrect:\nAnswer:'


def get_norm_probs(tokenizer, model, task_prompt, candidate_set, verbose=False):
    if verbose:
        print('**Generating probs for the following task:\n',
              task_prompt, '\n**Candidate set:', candidate_set)
    with torch.inference_mode():
        # get logits of last hidden state
        task_inputs = tokenizer(task_prompt, return_tensors="pt")
        next_token_logits = model(**task_inputs).logits[:, -1, :]
        filtered_next_token_logits = top_k_top_p_filtering(
            next_token_logits, top_k=50, top_p=0.9)
        if verbose:
            valid_token_num = torch.sum(~filtered_next_token_logits.isinf())
            sorted_ids = torch.argsort(-filtered_next_token_logits[0])[
                :valid_token_num]
            print('Nucleus sampled top-50 results:',
                  tokenizer.decode(sorted_ids))

        candidate_tokens = [tokenizer.encode(x)[-1]
                            for x in candidate_set]
        candidate_logits = torch.stack(
            [next_token_logits[0, x] for x in candidate_tokens])
        return candidate_logits.softmax(dim=-1).cpu().numpy()


def load_llama(path='meta-llama/Llama-2-7b-chat-hf', verbose=True):
    if verbose:
        print('loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(path)
    if verbose:
        print('loading model...')
    model = AutoModelForCausalLM.from_pretrained(
        path, device_map="auto", torch_dtype=torch.float16)
    return tokenizer, model


if __name__ == '__main__':
    Q = 'Where was Barack Obama born?'
    answers_text = ['Chicago', 'honolulu', 'Nairobi', 'NYC']
    answers_id = ['A', 'B', 'C', 'D']

    tokenizer, model = load_llama('D:/Codes/transformers/Llama-2-7b-hf')

    gen_probs = np.zeros((2, len(answers_text)))
    dis_probs = np.zeros((len(answers_text), 2))
    signal_mapping = {0: 'incorrect', 1: 'correct'}
    dis_candidate = ['False', 'True']

    # generator
    for signal in range(2):
        correct = signal_mapping[signal] == 'correct'
        gen_probs[signal] = get_norm_probs(tokenizer, model, task_prompt=gen_task(
            Q, answers_text, answers_id, correct=correct), candidate_set=answers_id, verbose=True)

    # discriminator
    for idx, ans_text in enumerate(answers_text):
        dis_probs[idx] = get_norm_probs(tokenizer, model, task_prompt=dis_task(
            Q, ans_text), candidate_set=dis_candidate, verbose=True)

    print('gen_probs:', gen_probs)
    print('dis_probs:', dis_probs)
    
    """
    gen_probs: 
    [[0.28316766 0.24989457 0.19461811 0.27231967]
    [0.38916737 0.33811447 0.15121414 0.12150399]]
    dis_probs: 
    [[0.4882834  0.5117166 ]
    [0.43206337 0.5679366 ]
    [0.54867351 0.45132652]
    [0.4921881  0.50781184]]
    """
