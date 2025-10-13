
import torch 
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.collate import ReformatModelAndTokForDiff, DiffusionCollator

model_name = "gpt2"
tokenizer_name = "gpt2"


reformatter = ReformatModelAndTokForDiff(model_name, tokenizer_name)
model, tokenizer = reformatter.get_model_tok()

def test_reformater():
    assert "[MASK]" in tokenizer.get_vocab(), "Special token [MASK] not found in tokenizer"
    print("Special token [MASK] successfully added")


    assert model.get_input_embeddings().weight.shape[0] == len(tokenizer), \
        "Model embedding size doesn't match tokenizer vocab size"
    print(f"Model embeddings resized to {len(tokenizer)} tokens")


    test_text = "Hello [MASK] world"
    encoded = tokenizer(test_text, return_tensors="pt")
    print(f"Successfully encoded text with [MASK] token: {encoded.input_ids}")

    print("\n")

def test_diffusion_collate():
    block_size = 16
    collate_fn = DiffusionCollator(tokenizer, block_size)

    sample_batch = [
        {"text": "The quick brown fox"},
        {"text": "jumps over the lazy dog"},
        {"text": "and runs away quickly"},
    ]

    batch_encodings = collate_fn(sample_batch)

    assert batch_encodings.input_ids.shape == (3, block_size), "Batch input shape mismatch"
    print(batch_encodings.input_ids)
    print(tokenizer.batch_decode(batch_encodings.input_ids, skip_special_tokens=False))

if __name__ == "__main__":
    test_reformater()
    test_diffusion_collate()