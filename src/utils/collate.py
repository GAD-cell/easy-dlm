import torch 
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

class ReformatModelAndTokForDiff():
    def __init__(self, model_name, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        diff_mask_token = {"additional_special_tokens": ["[MASK]"]}
        
        num_added_tokens = self.tokenizer.add_special_tokens(diff_mask_token)
        print(f"Added diffusion special tokens: {num_added_tokens}")
        
        self.model.resize_token_embeddings(len(self.tokenizer))

    def get_model_tok(self):
        return self.model, self.tokenizer

class DiffusionCollate():
    def __init__(self, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __call__(self, batch):
        texts = [item["text"] + self.tokenizer.eos_token for item in batch]

        input_encodings = self.tokenizer(
            texts,
            max_length=self.block_size,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        t = np.random.uniform(0, 1)
        mask  = torch.rand(input_encodings.input_ids.shape) < t

        return input_encodings


