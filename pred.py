import os
import numpy as np
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
    AutoTokenizer,
    
)
from datasets import load_dataset

from transformers.trainer_pt_utils import get_parameter_names
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
from transformers.generation import GenerationConfig

import logging

from femto import (StreamingDataset,QwenSmolForCausalLM,QwenSmolConfig, TrainConfig,display_model_summary,GenerateSampleCallback,num_parameters,SaveCheckpoint)

text ="Downloading dataset, from huggingface"


tokenizer = AutoTokenizer.from_pretrained("trainer_output/checkpoint-20000",local_files_only=True)
model = QwenSmolForCausalLM.from_pretrained("trainer_output/checkpoint-20000",local_files_only=True)
cfg =  GenerationConfig(
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



inputs = tokenizer(text, return_tensors="pt",truncation=True,padding=True).to(model.device)
with torch.inference_mode():
    outputs = model.generate(**inputs,generation_config=cfg)

print(outputs)
print(tokenizer.decode(outputs[0][0], skip_special_tokens=True))    
