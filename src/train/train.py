import argparse
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM
from datasets import load_dataset
import os
from muon import MuonClip, MuonConfig
from src.train.diffusion_trainer import A2DTrainer, A2DConfig
from src.utils.collate import ReformatModelAndTokForDiff, DiffusionCollator
from peft import LoraConfig, PeftModel, get_peft_model


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
        learning_rate=config.get('learning_rate', 5e-4),
        per_device_train_batch_size=config['per_device_train_batch_size'],
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
        max_grad_norm=config.get('max_grad_norm', None),
        lr_scheduler_type=config.get('lr_scheduler_type', 'linear'),
        warmup_steps=config.get('warmup_steps', 0),
        weight_decay=config.get('weight_decay', 0.0),
        fp16=config.get('fp16', False),
        remove_unused_columns=False,
        logging_steps=config.get('logging_steps', 10),
        save_steps=config.get('save_steps', 1000),
    )
    
    return training_config

def create_lora_config(config):
    lora_config = LoraConfig(
        r=config.get('lora_r', 8),
        lora_alpha=config.get('lora_alpha', 16),
        target_modules=config.get('lora_target_modules', ["q_proj", "v_proj", "o_proj", "k_proj", "up_proj", "down_proj"]),
        lora_dropout=config.get('lora_dropout', 0.1),
        bias="none",
        task_type="CAUSAL_LM"
    )
    return lora_config

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig
def get_llama_config(config, tokenizer) -> LlamaConfig:
    """
    Convert the model configuration to LlamaConfig.
    """
    return LlamaConfig(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        max_position_embeddings=config.max_position_embeddings,
        rope_theta=config.rope_theta,
        rms_norm_eps=config.rms_norm_eps,
        initializer_range=config.initializer_range,
        hidden_act=config.hidden_act,
        tie_word_embeddings=config.tie_word_embeddings,
    )


def main(config):
    """Main training function"""
    print("="*50)
    print("A2D Training Script")
    print("="*50)
    
    torch.manual_seed(config.get('seed', 42))

    
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    lora_config = create_lora_config(config) if config.get("use_lora", False) else None


    class ModelConfig_3M:
        hidden_size: int = 64
        intermediate_size: int = 256
        num_hidden_layers: int = 8
        num_attention_heads: int = 16
        hidden_act: str = "silu"
        block_size: int = 512
        max_position_embeddings: int = 2048
        initializer_range: float = 0.041666666666666664
        rms_norm_eps: float = 1e-6
        rope_theta: float = 10000.0
        attention_bias: bool = False
        tie_word_embeddings: bool = True

    llama_config = ModelConfig_3M()
    
    # use pretrained or not
    #model = LlamaForCausalLM(get_llama_config(llama_config,tokenizer)).to("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(config["model_name"])


    model, tokenizer = ReformatModelAndTokForDiff(model,
                                                tokenizer,
                                                lora_config=lora_config).get_model_tok()
    

    train_dataset, eval_dataset = load_and_preprocess_dataset(config, tokenizer)
    
    training_config = create_training_config(config)
    
    data_collator = DiffusionCollator(
        tokenizer=tokenizer,
        block_size=512,
    )
    

    model_config = AutoConfig.from_pretrained(config["model_name"])

    muon_config = MuonConfig(
        unified_lr = True,
        lr = config.get("learning_rate", 5e-4),
        lr_muon = config.get("lr_muon", 3e-4),
        lr_adam = config.get("lr_adam", 1e-8),

        muon_beta=0.95,
        muon_decay=config.get("weight_decay", 1e-4),
        ns_steps=5,

        adam_betas = (0.9, 0.95),
        adam_decay= config.get("weight_decay", 1e-4),
        adam_eps= 1e-10,

        enable_clipping= True,
        clipping_layers_mapping = {"q_proj":"q_proj","k_proj":"k_proj"} ,
        clipping_threshold= 50.0,
        clipping_alpha= 0.5,

        log_max_logits= False,
        log_dir= "" ,
        cans_ortho= False,
        estimate_lower_bound= False ,
    )

    optimizer = MuonClip(model, model_config, muon_config)

    print("\nInitializing A2DTrainer...")
    trainer = A2DTrainer(
        model=model,
        training_args=training_config,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(optimizer, None)  # Scheduler will be created by Trainer
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