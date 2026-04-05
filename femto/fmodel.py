import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional,Union,List,Tuple
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
# from transformers.modeling_outputs import MaskedLMOutput  # bert-like model


from .basics import TransformerBlock,RMSNorm,compute_rope_params
from .fconfig import QwenSmolConfig


class QwenSmolModel(PreTrainedModel):
    config_class = QwenSmolConfig

    def __init__(self, config:QwenSmolConfig):
        super().__init__(config)
        self.config:QwenSmolConfig = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.emb_dim,dtype=config.dtype)
        self.pre_norm = RMSNorm(config.emb_dim)
        self.trf_blocks  = nn.ModuleList([TransformerBlock(cfg=config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.emb_dim)
        # Reusable utilities
        if config.head_dim is None:
            head_dim = config.emb_dim // config.n_heads
        else:
            head_dim = config.head_dim
        cos, sin = compute_rope_params(head_dim=head_dim,theta_base=config.rope_base,context_length=config.context_length)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = config

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        # Get embeddings
        tok_embeds = self.embed_tokens(input_ids)

        # Create a causal mask
        x:torch.Tensor = tok_embeds
        num_tokens = x.shape[1]
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)
        # Forward through layers with None for cos, sin since they're computed in attention
        x = self.pre_norm(x)
        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)

        # Final Norm
        x = self.final_norm(x)

        return BaseModelOutputWithPast(last_hidden_state=x,past_key_values=None,hidden_states=None,attentions=None)
    


class QwenSmolForCausalLM(PreTrainedModel,GenerationMixin):
    config_class = QwenSmolConfig
    
    def __init__(self, config:QwenSmolConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model = QwenSmolModel(config)
        self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False, dtype=config.dtype)


        if config.tie_word_embeddings:
            self.out_head.weight =  self.model.embed_tokens.weight
            # https://discuss.huggingface.co/t/what-is-the-tie-word-embeddings-option-exactly-doing/8483
            self._tied_weights_keys = ['out_head.weight']
             

        self.main_input_name = "input_ids"
        

        self.config.is_decoder = True
        self.config.is_encoder_decoder = False
        self.gradient_checkpointing = True
        self.post_init()


    def tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.out_head, self.model.embed_tokens)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
        if self.config.tie_word_embeddings:
            self.out_head.weight =  self.model.embed_tokens.weight

    def get_output_embeddings(self):
        return self.out_head
    
    def set_output_embeddings(self, new_embeddings):
        self.out_head = new_embeddings
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.out_head.weight


    def eval(self):
        return super().train(False)
    

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # 
        x = outputs[0]

        # 
        logits = self.out_head(x)
        loss = None 
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
    

    def prepare_inputs_for_generation(
            self, 
            input_ids, 
            past_key_values = None, 
            attention_mask = None, 
            inputs_embeds = None, 
            cache_position = None, 
            **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:,-1:]
        # Create position_ids from attention_mask
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "use_cache": True,
        }       
    
    # def _reorder_cache(self, past_key_values, beam_idx):
    #     reordered_past = ()
    #     for layer_past in past_key_values:
    #         reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
    #     return reordered_past