
import torch 
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.collate import ReformatModelAndTokForDiff, DiffusionCollate

model_name = "gpt2"
tokenizer_name = "gpt2"


reformatter = ReformatModelAndTokForDiff(model_name, tokenizer_name)
model, tokenizer = reformatter.get_model_tok()

assert "[MASK]" in tokenizer.get_vocab(), "Special token [MASK] not found in tokenizer"
print("Special token [MASK] successfully added")


assert model.get_input_embeddings().weight.shape[0] == len(tokenizer), \
    "Model embedding size doesn't match tokenizer vocab size"
print(f"Model embeddings resized to {len(tokenizer)} tokens")


test_text = "Hello [MASK] world"
encoded = tokenizer(test_text, return_tensors="pt")
print(f"Successfully encoded text with [MASK] token: {encoded.input_ids}")

print("\n")