from typing import List, Dict
import torch
import gc

class HuggingFace:
    def __init__(self, model_name, model, tokenizer):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        # substitute '▁Sure' with 'Sure' (note: can lead to collisions for some target_tokens)
        self.pos_to_token_dict = {v: k.replace('▁', ' ') for k, v in self.tokenizer.get_vocab().items()}
        self.eos_token_ids = [self.tokenizer.eos_token_id]
        if 'llama3' in self.model_name.lower():
            self.eos_token_ids.append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        self.pad_token_id = self.tokenizer.pad_token_id

    def generate(self,
                 full_prompts_list: List[str],
                 max_n_tokens: int,
                 temperature: float,
                 top_p: float = 1.0) -> List[Dict]:
        if 'llama2' in self.model_name.lower():
            max_n_tokens += 1  # +1 to account for the first special token (id=29871) for llama2 models
        batch_size = len(full_prompts_list)
        vocab_size = len(self.tokenizer.get_vocab())
        inputs = self.tokenizer(full_prompts_list, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_n_tokens,
            do_sample=False if temperature == 0 else True,
            temperature=0,
            eos_token_id=self.eos_token_ids,
            pad_token_id=self.tokenizer.pad_token_id,  # added for Mistral
            top_p=top_p,
            output_scores=True,
            return_dict_in_generate=True,
        )
        output_ids = output.sequences
        # If the model is not an encoder-decoder type, slice off the input tokens
        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, input_ids.shape[1]:]
        if 'llama2' in self.model_name.lower():
            output_ids = output_ids[:, 1:]  # ignore the first special token (id=29871)

        generated_texts = self.tokenizer.batch_decode(output_ids)
        # output.scores: n_output_tokens x batch_size x vocab_size (can be counter-intuitive that batch_size doesn't go first)
        logprobs_tokens = [torch.nn.functional.log_softmax(output.scores[i_out_token], dim=-1).cpu().numpy()
                           for i_out_token in range(len(output.scores))]
        if 'llama2' in self.model_name.lower():
            logprobs_tokens = logprobs_tokens[1:]  # ignore the first special token (id=29871)

        logprob_dicts = [[{self.pos_to_token_dict[i_vocab]: logprobs_tokens[i_out_token][i_batch][i_vocab]
                          for i_vocab in range(vocab_size)}
                         for i_out_token in range(len(logprobs_tokens))
                         ] for i_batch in range(batch_size)]

        outputs = [{'text': generated_texts[i_batch],
                    'logprobs': logprob_dicts[i_batch],
                    'n_input_tokens': len(input_ids[i_batch][input_ids[i_batch] != 0]),  # don't count zero-token padding
                    'n_output_tokens': len(output_ids[i_batch]),
                    } for i_batch in range(batch_size)]

        for key in inputs:
            inputs[key].to('cpu')
        output_ids.to('cpu')
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()

        return outputs
