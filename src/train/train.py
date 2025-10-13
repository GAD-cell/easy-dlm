import argparse
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import os

from src.train.diffusion_trainer import A2DTrainer, A2DConfig
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

def create_training_config(config):
    training_config = A2DConfig(
        output_dir=config['output_dir'],
        num_train_epochs=config['num_train_epochs'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
        learning_rate=config['learning_rate'],
        fp16=config.get('fp16', False),
        remove_unused_columns=False,
        logging_steps=config.get('logging_steps', 10),
    )
    
    return training_config


def main(config):
    """Main training function"""
    print("="*50)
    print("A2D Training Script")
    print("="*50)
    
    torch.manual_seed(config.get('seed', 42))
    model, tokenizer = ReformatModelAndTokForDiff(model_name=config["model_name"],tokenizer_name=config["tokenizer_name"]).get_model_tok()
    
    train_dataset, eval_dataset = load_and_preprocess_dataset(config, tokenizer)
    
    training_config = create_training_config(config)
    
    data_collator = DiffusionCollator(
        tokenizer=tokenizer,
        block_size=config["block_size"],
    )
    

    print("\nInitializing A2DTrainer...")
    trainer = A2DTrainer(
        model=model,
        training_args=training_config,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    checkpoint = None
    if config.get('resume_from_checkpoint'):
        checkpoint = config['resume_from_checkpoint']
        print(f"Resuming from checkpoint: {checkpoint}")
    
    # Train
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50 + "\n")
    
    trainer.train(resume_from_checkpoint=checkpoint)
    
    print("\nSaving final model...")
    trainer.save_model(os.path.join(config['output_dir'], 'final_model'))
    tokenizer.save_pretrained(os.path.join(config['output_dir'], 'final_model'))
    
    if eval_dataset:
        print("\nFinal evaluation...")
        eval_results = trainer.evaluate()
        print(f"Final evaluation results: {eval_results}")
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train A2D model")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file (YAML)")
    args = parser.parse_args()
    
    print(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)