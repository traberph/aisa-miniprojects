from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
import os
import gc
import random
import string
import numpy as np
from glob import glob
from huggingface_hub import login
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

from hugging_face_model import HuggingFace
from prompts import FINAL_MESSAGE, FINAL_MESSAGE_ONESHOT

login(token=os.getenv("HF_TOKEN"))

SMOLLM3_3B_PATH = "HuggingFaceTB/SmolLM3-3B"
seed = 42
set_seed(seed)

def model_init():
    model = AutoModelForCausalLM.from_pretrained(
        SMOLLM3_3B_PATH,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True, device_map="cuda:0",
        token=os.getenv("HF_TOKEN"),
        trust_remote_code=True
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        SMOLLM3_3B_PATH,
        use_fast=False,
        token=os.getenv("HF_TOKEN")
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    return model, tokenizer

def extract_logprob(logprob_dict, target_token):
    logprobs = []
    if ' ' + target_token in logprob_dict:
        logprobs.append(logprob_dict[' ' + target_token])
    if target_token in logprob_dict:
        logprobs.append(logprob_dict[target_token])
    return max(logprobs) if logprobs else -np.inf

def insert_adv_string(msg, adv):
    return msg + adv

def early_stopping_condition(best_logprobs, target_model, logprob_dict, target_token, determinstic_jailbreak,
                              no_improvement_history=750, prob_start=0.02, no_improvement_threshold_prob=0.01):
    if determinstic_jailbreak and logprob_dict != {}:
        argmax_token = max(logprob_dict, key=logprob_dict.get)
        return argmax_token in [target_token, ' ' + target_token]

    if len(best_logprobs) == 0:
        return False

    best_logprob = best_logprobs[-1]
    if no_improvement_history < len(best_logprobs):
        prob_best = np.exp(best_logprobs[-1])
        prob_history = np.exp(best_logprobs[-no_improvement_history])
        no_sufficient_improvement_condition = prob_best - prob_history < no_improvement_threshold_prob
    else:
        no_sufficient_improvement_condition = False

    if isinstance(target_model.model, HuggingFace) and np.exp(best_logprob) > prob_start and no_sufficient_improvement_condition:
        return True
    if isinstance(target_model.model, HuggingFace) and np.exp(best_logprob) > 0.1:
        return True
    if np.exp(best_logprob) > 0.4:
        return True
    return False

def schedule_n_to_change_prob(max_n_to_change, prob, target_model):
    if isinstance(target_model.model, HuggingFace):
        if 0 <= prob <= 0.01:
            n_to_change = max_n_to_change
        elif 0.01 < prob <= 0.1:
            n_to_change = max_n_to_change // 2
        elif 0.1 < prob <= 1.0:
            n_to_change = max_n_to_change // 4
        else:
            raise ValueError(f'Wrong prob {prob}')
    else:
        if 0 <= prob <= 0.1:
            n_to_change = max_n_to_change
        elif 0.1 < prob <= 0.5:
            n_to_change = max_n_to_change // 2
        elif 0.5 < prob <= 1.0:
            n_to_change = max_n_to_change // 4
        else:
            raise ValueError(f'Wrong prob {prob}')
    return max(n_to_change, 1)

def schedule_n_to_change_fixed(max_n_to_change, it):
    if 0 < it <= 10:
        n_to_change = max_n_to_change
    elif 10 < it <= 25:
        n_to_change = max_n_to_change // 2
    elif 25 < it <= 50:
        n_to_change = max_n_to_change // 4
    elif 50 < it <= 100:
        n_to_change = max_n_to_change // 8
    elif 100 < it <= 500:
        n_to_change = max_n_to_change // 16
    else:
        n_to_change = max_n_to_change // 32
    return max(n_to_change, 1)

def run_mcmc(orig_msg, piece_idx, n_restarts=10, n_iters=10,
             n_tokens_adv=25, n_tokens_change_max=4, n_chars_change_max=0,
             target_max_n_tokens=150, determinstic_jailbreak=None,
             schedule_prob=True, schedule_n_to_change=None, tokenizer=None, lm=None, target_token=None):
    adv_init = ' !' * n_tokens_adv
    substitution_set = string.digits + string.ascii_letters + string.punctuation + ' '
    max_token_value = tokenizer.vocab_size

    global global_best_adv
    global global_best_logprob
    global global_best_response

    global_best_adv = adv_init
    global_best_logprob = -np.inf
    global_best_response = None

    for i_restart in range(n_restarts):
        early_stop_rs = False
        n_chars_change, n_tokens_change = 0, 4
        best_adv = adv = adv_init
        best_adv_tokens = adv_tokens = tokenizer.encode(adv_init)
        best_msg = msg = insert_adv_string(orig_msg, adv)
        best_logprob = -np.inf
        best_logprobs, logprob_dict = [], {}

        for it in range(1, n_iters + 1):
            if not early_stopping_condition(best_logprobs, lm, logprob_dict, target_token, None):
                output = lm.generate([msg], max_n_tokens=1, temperature=0)[0]
                logprob_dict = output['logprobs'][0]
                logprob = extract_logprob(logprob_dict, target_token)
            else:
                temperature = 0.0
                msg_early_stop = best_msg if determinstic_jailbreak else msg
                output = lm.generate([msg_early_stop], max_n_tokens=target_max_n_tokens, temperature=temperature)[0]
                logprob_dict = output['logprobs'][0]
                logprob = extract_logprob(logprob_dict, target_token)
                response_text = output['text']
                if response_text.strip().startswith(target_token):
                    global_best_response = response_text
                    early_stop_rs = True

            if logprob > best_logprob:
                best_logprob, best_msg, best_adv, best_adv_tokens = logprob, msg, adv, adv_tokens
            else:
                adv, adv_tokens = best_adv, best_adv_tokens
            best_logprobs.append(best_logprob)

            if early_stop_rs:
                break

            if n_chars_change_max > 0:
                if schedule_prob:
                    n_chars_change = schedule_n_to_change_prob(n_chars_change_max, np.exp(best_logprob), lm)
                elif schedule_n_to_change:
                    n_chars_change = schedule_n_to_change_fixed(n_chars_change_max, it)
                else:
                    n_chars_change = n_chars_change_max
                substitute_pos_start = random.choice(range(len(adv)))
                substitution = ''.join(random.choice(substitution_set) for _ in range(n_chars_change))
                adv = adv[:substitute_pos_start] + substitution + adv[substitute_pos_start + n_chars_change:]

            if n_tokens_change_max > 0:
                if schedule_prob:
                    n_tokens_change = schedule_n_to_change_prob(n_tokens_change_max, np.exp(best_logprob), lm)
                elif schedule_n_to_change:
                    n_tokens_change = schedule_n_to_change_fixed(n_tokens_change_max, it)
                else:
                    n_tokens_change = n_tokens_change_max
                substitute_pos_start = random.choice(range(len(adv_tokens)))
                substitution_tokens = np.random.randint(0, max_token_value, n_tokens_change).tolist()
                adv_tokens = (adv_tokens[:substitute_pos_start]
                              + substitution_tokens
                              + adv_tokens[substitute_pos_start + n_tokens_change:])
                adv = tokenizer.decode(adv_tokens).replace('<s>', '')  # R2D2 tokenizer inserts '<s>' at position 0

            msg = insert_adv_string(orig_msg, adv)

        if best_logprob > global_best_logprob:
            global_best_logprob = best_logprob
            global_best_adv = best_adv

        if early_stop_rs:
            break

    return global_best_adv, global_best_response, global_best_logprob, it

def preaparre_benchmark(
        path: str, 
        one_shot: bool = False
        ):
    try:
        adv_bench = load_dataset(
            "parquet"
        )
        harm_bench_cont = load_dataset(
            "walledai/HarmBench",
            "contextual"
        )
        harm_bench_copy = load_dataset(
            "walledai/HarmBench",
            "copyright"
        )
        harm_bench_std = load_dataset(
            "walledai/HarmBench",
            "standard"
        )
    except Exception as e:
        print(f"Could not laod datasets {e}, trying locally")
        all_path = glob(f"./{path}/HarmBench/*")

        adv_bench = load_dataset(
            "parquet",
            data_files=f"./{path}/AdvBench/train-00000-of-00001.parquet"
        )
        harm_bench_cont = load_dataset(
            "parquet",
            data_files=f"{all_path[0]}/train-00000-of-00001.parquet"
        )
        harm_bench_copy = load_dataset(
            "parquet",
            data_files=f"{all_path[1]}/train-00000-of-00001.parquet"
        )
        harm_bench_std = load_dataset(
            "parquet",
            data_files=f"{all_path[2]}/train-00000-of-00001.parquet"
        )
    

    adv_bench_df = adv_bench["train"].to_pandas()
    harm_bench_cont_df = harm_bench_cont["train"].to_pandas()
    harm_bench_copy_df = harm_bench_copy["train"].to_pandas()
    harm_bench_std_df = harm_bench_std["train"].to_pandas()

    adv_bench_df_01 = adv_bench_df.sample(frac=0.05, random_state=42)
    harm_bench_cont_df_01 = harm_bench_cont_df.sample(frac=0.05, random_state=42)
    harm_bench_copy_df_01 = harm_bench_copy_df.sample(frac=0.05, random_state=42)
    harm_bench_std_df_01 = harm_bench_std_df.sample(frac=0.05, random_state=42)

    all_queries = [
        adv_bench_df_01["prompt"].tolist()
        +harm_bench_cont_df_01["prompt"].tolist()
        +harm_bench_copy_df_01["prompt"].tolist()
        +harm_bench_std_df_01["prompt"].tolist()
    ]

    if one_shot:
        adv_bench_df_rest = adv_bench_df.drop(adv_bench_df_01.index)
        example_query, example_response = adv_bench_df_rest["prompt"][0], adv_bench_df_rest["target"][0]

        return all_queries, example_query, example_response
    
    return all_queries

if __name__ == "__main__":
    model, tokenizer = model_init()

    lm = HuggingFace("smollm3-3b", model, tokenizer)

    target_token = "Sure"
    path = "datasets"
    one_shot=False

    if one_shot:
        all_queries, example_query, example_response = preaparre_benchmark(
            path=path,
            one_shot=one_shot
        )
    else:
        all_queries = preaparre_benchmark(
            path=path
        )

    all_check = []
    for query in tqdm(all_queries[0], total=len(all_queries[0])):
        if one_shot:
            message_prompt = FINAL_MESSAGE_ONESHOT[0].format(
                target_str=query,
                one_shot_query=example_query,
                one_shot_answer=example_response,
                )
        else:
            message_prompt = FINAL_MESSAGE[0].format(
                target_str=query,
                one_shot_query=example_query,
                one_shot_answer=example_response,
                )
            
        st_adv, best_response, best_logprob, it = run_mcmc(
            message_prompt, 
            0,
            tokenizer=tokenizer,
            lm=lm,
            target_token=target_token
            )
        all_check.append({
            "query":query,
            "attack":st_adv,
            "best_response":best_response,
            "best_logprob":best_logprob,
            "it":it
        })

    df = pd.DataFrame(all_check)

    df.to_csv("./results.csv", index=False)
