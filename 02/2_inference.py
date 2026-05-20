import argparse
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch
import hashlib
from transformers import LogitsProcessor, LogitsProcessorList


class GreenListWatermarkProcessor(LogitsProcessor):
    def __init__(self, vocab_size, gamma=0.5, delta=2.0, k=1):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.k = k
        self.green_size = int(gamma * vocab_size)
        

    def __call__(self, input_ids, logits):
        for i in range(input_ids.shape[0]):
            last_k = input_ids[i, -self.k:].tolist()
            seed = abs(hash(tuple(last_k))) % (2**32)

            gen = torch.Generator(device="cpu").manual_seed(seed)
            perm = torch.randperm(self.vocab_size, generator=gen)
            green_ids = perm[:self.green_size].to(logits.device)

            logits[i, green_ids] += self.delta
        return logits

def generate_response(
        query: str,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        thinking: bool = False,
        context: str = None,
        processor: GreenListWatermarkProcessor = None
        ):
    
    response_dct = {}
    messages = [
        {
            "role": "user", 
            "content": query
            # "content": f"Provide the response with the given {context} {query}"
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    if processor:
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,
            logits_processor=LogitsProcessorList([processor]),
            do_sample=True,
            temperature=0.7
        )
    else:
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7
        )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    if thinking:
        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        response_dct["thinking_content"] = thinking_content
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    response_dct["content"] = content

    return response_dct

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--processor', action='store_true', help='A boolean flag')
    args = parser.parse_args()
    processor = None

    ds = datasets.load_dataset("databricks/databricks-dolly-15k")
    df = ds['train'].to_pandas()
    df_sample = df.sample(n=100, random_state=42)

    model_name = "Qwen/Qwen3-4B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    if args.processor:
        processor = GreenListWatermarkProcessor(vocab_size=tokenizer.vocab_size, gamma=0.5, delta=2.0, k=1,)

    queries = df_sample["instruction"].tolist()
    context_all = df_sample["context"].tolist()

    responses = []
    for query, context in tqdm(zip(queries, context_all), total=len(queries)):
        responses.append(
            generate_response(
                query=query,
                tokenizer=tokenizer,
                model=model,
                thinking=False,
                context=context,
                processor=processor
            )
        )

    generated_output = [resp["content"] for resp in responses]
    
    df_sample["generated_response"] = generated_output

    df_sample.to_csv("./generated_responses_100_watermaked_no_context.csv", index=False)
