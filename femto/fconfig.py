from transformers import PretrainedConfig
from typing import Optional
from dataclasses import dataclass


@dataclass
class TrainConfig:
    tokenizer_name:str 
    batch_size:int
    max_seq_len:int  # context_lenght
    lr:float
    warmup_steps:int
    training_step:int
    checkpoint_interval:int 
    sample_generation_interval:int 
    checkpnt_dir:str="trainer_output/last.pth"
    


class QwenSmolConfig(PretrainedConfig):
    model_type = "qwensmol"

    def __init__(
        # initalize with default config 'model.__class__' expected behaviour while doing model.save_pretrained()
        self,
        vocab_size:int=49152,
        context_length:int=512,
        emb_dim:int=768,
        n_heads:int=8,
        n_layers:int=3,
        hidden_dim:int=384,
        head_dim:int=48,
        qk_norm:bool=True,
        n_kv_groups:int=4,
        rope_base:int=10000,

        num_experts:Optional[int]=4,
        num_experts_per_tok:Optional[int]=4,
        moe_intermediate_size:Optional[int]=12,
        tie_word_embeddings: bool=True,

        tokenizer_class: Optional[str] = None,
        prefix: Optional[str] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        sep_token_id: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        torchscript:bool = False,
        
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.qk_norm = qk_norm
        self.n_kv_groups = n_kv_groups
        self.rope_base = rope_base
        self.num_experts=num_experts
        self.num_experts_per_tok=num_experts_per_tok
        self.moe_intermediate_size=moe_intermediate_size
        self.tie_word_embeddings = tie_word_embeddings
        self.torchscript = torchscript

        # we already know it's decoder only model
        self.is_encoder_decoder = False
        self.is_decoder = True
        # tokenizer kwargs
        self.tokenizer_class = tokenizer_class
        self.prefix = prefix
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.sep_token_id = sep_token_id
        self.decoder_start_token_id = decoder_start_token_id

        # generation expects num_hidden_layers so mapped it accordingly
        self.num_hidden_layers = self.n_layers


# if __name__=="__main__":
# # 300m params 
#       cfg = QwenSmolConfig(
#           vocab_size= tokenizer.vocab_size,
#           context_length=5120,
#           emb_dim=768,
#           n_heads = 8,
#           n_layers = 9,
#           hidden_dim = 1536,
#           head_dim = 48,
#           qk_norm = True,
#           n_kv_groups = 4 ,
#           rope_base = 100000,
#           num_experts = 32,
#           num_experts_per_tok =  4,
#           moe_intermediate_size = 384,
#           tie_word_embeddings=True
#       )