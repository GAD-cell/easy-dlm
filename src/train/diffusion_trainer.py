from transformers import Trainer, TrainingArguments
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math

def selective_log_softmax(logits, index) -> torch.Tensor:
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficient approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


class A2DConfig(TrainingArguments):
    pass


class A2DTrainer(Trainer):
    def __init__(self, 
    model: nn.Module, 
    training_args, 
    data_collator, 
    train_dataset, 
    eval_dataset=None,
    optimizers=(None,None)):

        self.model = model
        self.args = training_args

        super().__init__(model=model, 
                        args=training_args, 
                        data_collator=data_collator, 
                        train_dataset=train_dataset, 
                        eval_dataset=eval_dataset,
                        optimizers=optimizers)

    def _prepare_inputs(self, batch):

        output = {
            "input_ids": batch["input_ids"].to(self.args.device),
            "attention_mask": batch["attention_mask"].to(self.args.device),
            "labels": batch["labels"].to(self.args.device),
            "diffusion_masks": batch["diffusion_masks"].to(self.args.device),
            "t": batch["t"].to(self.args.device),
                }
        return output
    
    def compute_loss(self, model, inputs):
        input_ids, attention_mask,  = inputs["input_ids"], inputs["attention_mask"]
        labels, diffusion_masks = inputs["labels"], inputs["diffusion_masks"]
        t = inputs["t"]

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        # translate diffusion masks to predict only if next token is masked
        trans_masks = torch.cat([
            diffusion_masks[:, 1:], 
            torch.zeros(diffusion_masks.shape[0], 1, dtype=torch.bool, device=diffusion_masks.device)
        ], dim=1)
        
        logps = selective_log_softmax(logits, labels)
        loss = - (logps[trans_masks] / t).mean()

        return loss

    def training_step(self, model, inputs, num_items_in_batch=None) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Compute loss
        loss = self.compute_loss(model, inputs)
        
        # Scale loss for gradient accumulation
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        # Backward pass (simple)
        loss.backward()

        return loss.detach()

    def train(self, resume_from_checkpoint=False):
        """
        Main training loop.
        """

        
        # Setup
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        num_train_epochs = self.args.num_train_epochs
        max_steps = math.ceil(num_train_epochs * num_update_steps_per_epoch)
        
        self.model.to(self.args.device)
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        
        # Setup mixed precision scaler si FP16
        scaler = None
        if self.args.fp16 and torch.cuda.is_available():
            from torch.amp import GradScaler
            scaler = GradScaler('cuda')
            print("FP16 training enabled with GradScaler")
        
        # Training loop
        total_loss = 0
        self.global_step = 0
        
        for epoch in range(int(num_train_epochs)):
            epoch_loss = 0
            self.model.train()
            
            for step, batch in enumerate(train_dataloader):
                inputs = self._prepare_inputs(batch)
                
                if self.args.fp16 and scaler is not None:
                    from torch.amp import autocast
                    with autocast('cuda'):
                        loss = self.compute_loss(self.model, inputs)
                        
                        # Scale loss for gradient accumulation
                        if self.args.gradient_accumulation_steps > 1:
                            loss = loss / self.args.gradient_accumulation_steps
                    
                    scaler.scale(loss).backward()
                else:
                    loss = self.compute_loss(self.model, inputs)
                    

                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps
                    
                    loss.backward()
                
                total_loss += loss.item()
                epoch_loss += loss.item()
                
                # Gradient accumulation
                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    # Clip gradients
                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                        if scaler is not None:
                            # Unscale before clipping
                            scaler.unscale_(self.optimizer)
                        
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.args.max_grad_norm,
                        )
                    
                    # Optimizer step
                    if scaler is not None:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()
                    
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    
                    # Logging
                    if self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0:
                        avg_loss = total_loss / self.args.logging_steps
                        print(f"Epoch {epoch} | Step {self.global_step} | Loss: {avg_loss:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.2e}")
                        total_loss = 0
            

            avg_epoch_loss = epoch_loss / len(train_dataloader)
            print(f"Epoch {epoch} completed | Average Loss: {avg_epoch_loss:.4f}")
            
            # Evaluation
            if self.args.eval_strategy == "epoch" and self.eval_dataset is not None:
                eval_results = self.evaluate()
                print(f"Evaluation results: {eval_results}")
            
            # Save checkpoint
            if self.args.save_strategy == "epoch":
                self.save_model(f"{self.args.output_dir}/checkpoint-epoch-{epoch}")
        
        print("Training completed!")
        return self.model