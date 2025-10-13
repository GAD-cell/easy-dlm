from transformers import Trainer

class A2DTrainer(Trainer):
    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        super().__init__(model=model, args=args, tokenizer=tokenizer)
    def _prepare_inputs(self, batch)
        pass
    
    def compute_loss(self, model,inputs):

        
        loss = #compute loss
        return loss

    def training_step(self, model, inputs, num_items_in_batch=None) -> torch.Tensor:
        
        inputs = self._prepare_inputs(inputs)
        model.train()
        loss = self.compute_loss(model,inputs)


        return loss.detach()
