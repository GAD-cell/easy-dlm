import argparse
import yaml
import torch
from datasets import load_dataset
from src.utils.collate import ReformatModelAndTokForDiff, DiffusionCollator


def load_and_preprocess_dataset(config, tokenizer):
    """Load dataset WITHOUT tokenization (done by collator)"""
    print(f"Loading dataset: {config['dataset_name']}")
    
    if 'dataset_path' in config and config['dataset_path']:
        dataset = load_dataset(config["dataset_path"], config.get("dataset_config_name"), cache_dir=config.get("cache_dir"))
    else:
        dataset = load_dataset(config['dataset_name'], cache_dir=config.get("cache_dir"))
    
    train_dataset = dataset['train']
    eval_dataset = dataset.get('validation') or dataset.get('test')
    
    print(f"Train samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval samples: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
def main(config):
    model = AutoModelForCausalLM.from_pretrained(config["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    model, tokenizer = ReformatModelAndTokForDiff(model,
                                                tokenizer).get_model_tok()

    data_collator = DiffusionCollator(
    tokenizer=tokenizer,
    block_size=512,
    )
    sub_block_size = 16

    _, eval_dataset = load_and_preprocess_dataset(config, tokenizer)

    eval_sample = eval_dataset[0]
    without_resampling1(eval_sample, data_collator, model, tokenizer)

    # eval_sample = {"test":"Il était une fois "}
    # without_resampling2(eval_sample, model, tokenizer, sub_block_size=sub_block_size)
    # test_resampling(eval_sample, model, tokenizer)

def without_resampling1(eval_sample,data_collator, model, tokenizer):
    batch = data_collator([eval_sample])
    print(f"Batch after collate: {tokenizer.decode(batch['input_ids'][0][batch['attention_mask'][0]==1])}")
    output = model(**batch)
    print(f"Model output logits shape: {output.logits.shape}")
    print(tokenizer.decode(output.logits[0].argmax(dim=-1)))

def without_resampling2(eval_sample, model, tokenizer, sub_block_size=16):

    encodings = tokenizer(eval_sample["test"], return_tensors="pt")
    mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")
    encodings["input_ids"] = torch.cat([encodings["input_ids"], torch.tensor([[mask_token_id]*sub_block_size])], dim=1)
    encodings["attention_mask"] = torch.ones(1, encodings["input_ids"].shape[1], dtype=torch.bool)

    output = model(**encodings)
    print(f"Model output logits shape: {output.logits.shape}")
    print(tokenizer.decode(output.logits[0].argmax(dim=-1)))


from src.inference.sampling import RND1GenerationConfig, RND1GenerationMixin
def test_resampling(inputs,model,tokenizer):
    encodings = tokenizer(inputs["test"], return_tensors="pt")
    generation_config = RND1GenerationConfig(mask_token_id=tokenizer.convert_tokens_to_ids("[MASK]"),
                                            pad_token_id=tokenizer.pad_token_id,
                                            eos_token_id=tokenizer.eos_token_id,)

    generator = RND1GenerationMixin()
    model = model.to("cuda:0")
    output = generator.generate(model, encodings["input_ids"], generation_config=generation_config)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train A2D model")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file (YAML)")
    args = parser.parse_args()
    
    print(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)