import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM, DefaultDataCollator

class ReformatModelAndTokForDiff():
    def __init__(self, model_name, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        diff_mask_token = {"additional_special_tokens": ["[MASK]"]}  
        num_added_tokens = self.tokenizer.add_special_tokens(diff_mask_token)
        print(f"Added diffusion special tokens: {num_added_tokens}")

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.tokenizer.padding_side = "right"

        self.model.resize_token_embeddings(len(self.tokenizer))

    def get_model_tok(self):
        return self.model, self.tokenizer


# class to convert causal attention into full attention
class CausalToFullAttention(torch.nn.Module):
    pass


class DiffusionCollator(DefaultDataCollator):
    def __init__(self, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.diffusion_mask_id = self.tokenizer.convert_tokens_to_ids("[MASK]")
        self.pad_id = self.tokenizer.pad_token_id
        self.t_eps = 1e-8

    def __call__(self, batch):
        texts = [item["text"] + self.tokenizer.eos_token for item in batch]

        input_encodings = self.tokenizer(
            texts,
            max_length=self.block_size,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        input_encodings["labels"] = input_encodings.input_ids.clone() # no shift, it's not a causal task

        t = torch.rand(input_encodings.input_ids.shape[0], 1)  # per sequence masking
        mask  = (torch.rand(input_encodings.input_ids.shape) < t) & (input_encodings.input_ids != self.pad_id)
        input_encodings.input_ids[mask] = self.diffusion_mask_id
        input_encodings["diffusion_masks"] = mask
        input_encodings["t"] = t + self.t_eps  # avoid div by zero
        return input_encodings


