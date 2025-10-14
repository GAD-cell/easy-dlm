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

def main(config):
    model, tokenizer = ReformatModelAndTokForDiff(model_name=config["model_name"],
                                                tokenizer_name=config["tokenizer_name"]).get_model_tok()

    data_collator = DiffusionCollator(
    tokenizer=tokenizer,
    block_size=512,
    )

    _, eval_dataset = load_and_preprocess_dataset(config, tokenizer)

    eval_sample = eval_dataset[0]

    without_resampling(eval_sample, data_collator, model, tokenizer)


def without_resampling(eval_sample,data_collator, model, tokenizer):
    batch = data_collator([eval_sample])
    print(f"Batch after collate: {tokenizer.decode(batch['input_ids'][0][batch['attention_mask'][0]==1])}")
    output = model(**batch)
    print(f"Model output logits shape: {output.logits.shape}")
    print(tokenizer.decode(output.logits[0].argmax(dim=-1)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train A2D model")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file (YAML)")
    args = parser.parse_args()
    
    print(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)