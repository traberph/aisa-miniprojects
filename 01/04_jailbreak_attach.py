from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
import os
import gc
import random
import string
import numpy as np
from typing import List, Dict
from huggingface_hub import login
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