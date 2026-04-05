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

from transformers.trainer_pt_utils import get_parameter_names
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
from transformers.generation import GenerationConfig

import logging

from femto import (StreamingDataset,QwenSmolForCausalLM,QwenSmolConfig, TrainConfig,display_model_summary,GenerateSampleCallback,SaveCheckpoint,setup_logger)



os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

setup_logger()



tconfig = TrainConfig(
    tokenizer_name="HuggingFaceTB/SmolLM-135M",
    training_step=50000,
    batch_size=32,
    max_seq_len=100,
    lr=1e-3,
    warmup_steps=0.3,
    checkpoint_interval=20000,
    sample_generation_interval=10000,

)

femto_tokenizer = AutoTokenizer.from_pretrained(tconfig.tokenizer_name, use_fast=True)
logging.info("*" * 10 + " Tokenizer loaded successfully!! " + "*" * 10)


# 300m params [exclude optimizer & scheduler]
model_cfg = QwenSmolConfig(
    vocab_size=femto_tokenizer.vocab_size,
    context_length=tconfig.max_seq_len,
    emb_dim=768,
    n_heads=8,
    n_layers=3,
    hidden_dim=384,
    head_dim=48,
    qk_norm=True,
    n_kv_groups=4,
    rope_base=1000,
    num_experts=4,
    num_experts_per_tok=4,
    moe_intermediate_size=192,
    tie_word_embeddings=True,
    torchscript=True,
)
logging.info("*" * 10 + " Config loaded successfully!! " + "*" * 10)

if femto_tokenizer.pad_token is None:
    femto_tokenizer.pad_token = femto_tokenizer.eos_token
    femto_tokenizer.pad_token_id = femto_tokenizer.eos_token_id

# Also ensure other special tokens are set
if femto_tokenizer.bos_token is None:
    femto_tokenizer.bos_token = femto_tokenizer.eos_token
    femto_tokenizer.bos_token_id = femto_tokenizer.eos_token_id

tconfig.pad_token_id = femto_tokenizer.pad_token_id
tconfig.bos_token_id = femto_tokenizer.bos_token_id
tconfig.eos_token_id = femto_tokenizer.eos_token_id


logging.info("*" * 10 + "Initializing model," + "*" * 10)
model = QwenSmolForCausalLM(config=model_cfg).cuda()


logging.info("*" * 10 + "Initializing dataset and dataloader," + "*" * 10)
train_dataset = StreamingDataset(femto_tokenizer, block_size=tconfig.max_seq_len)


logging.info("Dataset and dataloader initialized successfully")

# # # iterable_ds = iter(train_dataset)
# # # # rich.print(next(iterable_ds))


logging.info("Initializing optimizer and scheduler, Please wait...")
# Optimizer
    # Split weights in two groups, one with weight decay and the other not.
forbidden_name_patterns = [r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm], forbidden_layer_names=forbidden_name_patterns)
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if n in decay_parameters and p.requires_grad],
        "weight_decay": 0.1,
    },
    {
        "params": [p for n, p in model.named_parameters() if n not in decay_parameters and p.requires_grad],
        "weight_decay": 0.0,
    },
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=tconfig.lr)

# optimizer = torch.optim.AdamW(model.parameters(), lr=tconfig.lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=tconfig.warmup_steps,
    num_training_steps=tconfig.training_step,
)



with torch.amp.autocast(enabled=True,dtype=torch.float16,device_type='cuda'):
    for name, param in model.named_parameters():
        print(name, param.dtype)

display_model_summary(model=model)

#  Create an efficient collator which dynamically pads  End-of-sequence as the padding token and mlm=False will use the inputs as labels, shifted to the right by one element
text = "Once upon a time, thre lived a ghost"
callbks = [
    GenerateSampleCallback(tokenizer=femto_tokenizer,prompt=text,every_n_steps=tconfig.sample_generation_interval),
    # SaveCheckpoint(path=tconfig.checkpnt_dir,save_n_steps=tconfig.checkpoint_interval)
]


trainer = Trainer(
    model=model,
    callbacks=callbks,
    args=TrainingArguments(
        ## Training
        do_train=True,
        do_eval=False,
        do_predict=False,
        eval_strategy='no',

        ## dataloader
        dataloader_num_workers=0,  #  `0` workers good for streaming

        ## Log
        # trackio_space_id= "trackio",
        project="deepsmol",
        report_to="tensorboard",
        # logging_dir=".logs",
        logging_steps=100,
        logging_nan_inf_filter=True,


        ## save model
        # save_total_limit=0,
        # save_strategy="best",
        # overwrite_output_dir=False,
        # load_best_model_at_end=True,
        save_steps=tconfig.checkpoint_interval,

        ## steps
        max_steps=tconfig.training_step,
        
        
        
        
        learning_rate=tconfig.lr,
        fp16=True,
        fp16_opt_level="O3",
        
        
        jit_mode_eval=True,
        include_num_input_tokens_seen=True,
        include_for_metrics=True,
        include_tokens_per_second=True,
        torch_compile=True,
        disable_tqdm=False,
        skip_memory_metrics=False,
        
        
    ),

    ########################### Uncomment in fdataset.py #########################################################
    #       # # Train Iterable Dataset Over Trainer API
    #       # yield self.tokenizer(text.strip(),truncation=True,max_length=self.block_size,padding=False)
    #       DataCollatorForLanguageModeling Expects in that way
    ####################################################################################
    train_dataset=train_dataset,
    data_collator=DataCollatorForLanguageModeling(femto_tokenizer, mlm=False),
    optimizers=(optimizer, scheduler),
)

# issue: dataset>=4.3 switch to datasets==3.*
# Fatal Python error: PyGILState_Release: thread state 0x72cc741bc290 must be current when releasing Python runtime state: finalizing (tstate=0x00000000009be910)
trainer.train()


gen_config = GenerationConfig(
    max_new_tokens=5,
    # beam-searh decoding
    do_sample=True,
    num_beams=4,
    early_stopping=True,
    top_k=10,
    top_p=0.9,
    eos_token_id=femto_tokenizer.eos_token_id,
    forced_eos_token_id=femto_tokenizer.eos_token_id,
    return_dict_in_generate=True,
    output_scores=True,
)


message = "As a human we live in"

inputs = femto_tokenizer(message, return_tensors="pt",truncation=True,padding=False,padding_side='right',).to('cuda')

outputs = model.generate(
    **inputs, 
    generation_config=gen_config,
    # logits_processor= RepetitionPenaltyLogitsProcessor(penalty=1.1,prompt_ignore_length=inputs["input_ids"].shape[-1])
    )

transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)

input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]

generated_tokens = outputs.sequences[:, input_length:]

logging.info("PREDICT:: "+ message+ ''.join([femto_tokenizer.decode([_]) for _ in generated_tokens[0]]))

for tok, score in zip(generated_tokens[0], transition_scores[0]):
    # | token | token string | log probability | probability
    logging.info(f"| {tok:5d} | {femto_tokenizer.decode(tok):8s} | {score.cpu().numpy():.3f} | {np.exp(score.cpu().numpy()):.2%}")