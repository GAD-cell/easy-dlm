from transformers import Trainer, TrainingArguments
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
from dataclasses import field


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
        self.is_pretrained = True
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
    
    def compute_loss(self, model, inputs,num_items_in_batch=None) -> torch.Tensor:
        input_ids, attention_mask,  = inputs["input_ids"], inputs["attention_mask"]
        labels, diffusion_masks = inputs["labels"], inputs["diffusion_masks"]
        t = inputs["t"]

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        
        logps = selective_log_softmax(logits, labels)
        if self.is_pretrained:
            # translate diffusion masks to predict only if next token is masked
            trans_masks = torch.cat([
                diffusion_masks[:, 1:], 
                torch.zeros(diffusion_masks.shape[0], 1, dtype=torch.bool, device=diffusion_masks.device)
            ], dim=1)
            loss = - (logps[trans_masks] / (1+t)).mean()
        
        else:
            loss = - (logps[diffusion_masks]).mean()

        loss = loss / self.args.gradient_accumulation_steps
        #self.log({"train_loss": loss.detach().cpu().item()})
        return loss