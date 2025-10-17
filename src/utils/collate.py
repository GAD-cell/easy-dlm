import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM, DefaultDataCollator
import transformers


def full_attention_causal_mask(self, 
                        attention_mask, 
                        input_tensor, 
                        cache_position=None, 
                        past_key_values=None, 
                        output_attentions=False):
      
    batch_size, seq_length = input_tensor.shape[:2]
    mask_queries = attention_mask[:, None, :, None].bool()  # (batch, 1, seq_len, 1)
    mask_keys = attention_mask[:, None, None, :].bool()     # (batch, 1, 1, seq_len)
    
    full_mask = mask_queries & mask_keys  # (batch, 1, seq_len, seq_len)
    
    return full_mask

class ReformatModelAndTokForDiff():
    def __init__(self, model, tokenizer, lora_config=None):
        self.tokenizer = tokenizer
        self.model = model
        

        diff_mask_token = {"additional_special_tokens": ["[MASK]"]}  
        num_added_tokens = self.tokenizer.add_special_tokens(diff_mask_token)
        print(f"Added diffusion special tokens: {num_added_tokens}")

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.tokenizer.padding_side = "right"
 
        
        
        # if lora_config:
        #     from peft import get_peft_model
        #     self.model = get_peft_model(self.model, lora_config)
        #     print("Converted model to LoRA model")
        #     print(self.model.print_trainable_parameters())

        self._patch_attention_mechanism()
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    def _patch_attention_mechanism(self):
        """Patch the attention mechanism to use full attention"""

        if hasattr(self.model, 'transformer'):
            base_model = self.model.transformer
        elif hasattr(self.model, 'model'):
            base_model = self.model.model
        else:
            base_model = self.model
        
        base_model.__class__._update_causal_mask = full_attention_causal_mask
        print(f"Patched {base_model.__class__.__name__} to use full attention")

    def get_model_tok(self):
        return self.model, self.tokenizer


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
        
        input_encodings["labels"] = input_encodings.input_ids.clone() 
        # input_encodings["labels"][:,:-1] = input_encodings["labels"][:,1:]  # shift left
        # input_encodings["labels"][:,-1] = self.tokenizer.pad_token_id

        t = torch.rand(input_encodings.input_ids.shape[0], 1)  # per sequence masking

        mask  = (torch.rand(input_encodings.input_ids.shape) < t) & (input_encodings.input_ids != self.pad_id)
        input_encodings.input_ids[mask] = self.diffusion_mask_id
        input_encodings["diffusion_masks"] = mask
        input_encodings["t"] = t + self.t_eps  # avoid div by zero
        return input_encodings


