import torch 
import numpy as np

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


