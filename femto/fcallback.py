import torch
from transformers.trainer_callback import TrainerCallback
from .fhelper import save_checkpoint
import logging
from transformers.generation import GenerationConfig

logger = logging.getLogger(__name__)


class GenerateSampleCallback(TrainerCallback):
    def __init__(
        self, tokenizer, prompt, every_n_steps: int , max_new_tokens: int = 50
    ):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.every_n_steps = every_n_steps
        self.max_new_tokens = max_new_tokens
        self._cfg =  GenerationConfig(
                        max_new_tokens=5,
                        # beam-searh decoding
                        do_sample=True,
                        num_beams=4,
                        early_stopping=True,
                        top_k=10,
                        top_p=0.9,
                        eos_token_id=tokenizer.eos_token_id,
                        forced_eos_token_id=tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )



    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every_n_steps == 0 and state.global_step > 0:
            model = kwargs.get("model")
            inputs = self.tokenizer(self.prompt, return_tensors="pt",truncation=True).to(model.device)
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=50)
            logger.info(f"{state.global_step}::" +self.tokenizer.decode(outputs[0], skip_special_tokens=True))
            


class SaveCheckpoint(TrainerCallback):
    def __init__(
        self, path, save_n_steps: int,
    ):
        self.path = path 
        self.every_n_steps = save_n_steps

            
    def on_train_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        optimizer = kwargs.get("optimizer")
        scheduler = kwargs.get('scheduler')
        step = state.global_step
        save_checkpoint(model=model,scheduler=scheduler,optimizer=optimizer,step=step,path=self.path)

            