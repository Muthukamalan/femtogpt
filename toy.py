"""
A decoder-only Transformer implementation in Pytorch from scratch.
The code is inspired from @karpathy's "nanoGPT" implementation.
The code is organized as follows:
- `Head`: Implements one head of self-attention.
- `MultiHeadAttention`: Implements multiple heads of self-attention in parallel.
- `FeedForward`: Implements the feed forward layer. Uses a simple 2-layer MLP with 
   GELU non-linearity in between.
- `Block`: Implements a block of attention, which consists of multi-head attention 
   and feed forward layers, along with residual connections and RMS normalization.
- `Transformer`: Implements the overall Transformer model, which consists of token 
   and positional embeddings, multiple blocks of attention, and a final linear layer 
   to produce logits for the next token prediction. 
   It also includes a `generate` method for autoregressive text generation.
Input parameters for the `Transformer` class include:
- `vocab_size`: The size of the vocabulary (number of unique tokens).
- `embedding_dimension`: The dimension of the token and positional embeddings.
- `num_heads`: The number of heads in the multi-head attention mechanism.
- `block_size`: The maximum context length (number of tokens the model can attend to).
- `num_layers`: The number of blocks of attention in the model.
- `kv_cache_enabled`: If KV-Cache is enabled at the inference time.
"""

# [Tensor Dim](https://huggingface.co/blog/not-lain/tensor-dims)
# [KV Explain](https://huggingface.co/blog/not-lain/kv-caching)

import torch.nn as nn
import torch.nn.functional as F
import torch


class Head(nn.Module):
    """One Head of Self-Attention"""

    def __init__(self, head_size, embedding_dimension, block_size):
        super().__init__()
        self.key = nn.Linear(embedding_dimension, head_size, bias=False)
        self.query = nn.Linear(embedding_dimension, head_size, bias=False)
        self.value = nn.Linear(embedding_dimension, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        
        # KV Cache buffers
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.current_pos = 0

    def forward(self, x, kv_cache_enabled=False):
        B, T, C = x.shape
        k_current = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v_current = self.value(x)  # (B, T, head_size)

        if kv_cache_enabled:
            if self.cache_k is None:
                self.cache_k, self.cache_v = k_current, v_current
            else:
                self.cache_k = torch.cat([self.cache_k, k_current], dim=1)
                self.cache_v = torch.cat([self.cache_v, v_current], dim=1)
            k, v = self.cache_k, self.cache_v
        else:
            k, v = k_current, v_current

        # compute attention scores ("affinities")
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * (C**-0.5)

        # Decoder only Transformer: Allows the token at their
        # current position to attend to the past tokens, but not the future tokens.
        if kv_cache_enabled:
            wei = wei.masked_fill(
                self.tril[self.current_pos:self.current_pos + q.shape[-2], :k.shape[-2]] == 0,
                float("-inf")
            )  # B,T1,T
            self.current_pos += q.shape[-2]
        else:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        out = wei @ v  # (B, T, head_size)
        return out
    
    def reset_kv_cache(self):
        self.cache_k,self.cache_v  = None, None
        self.current_pos = 0


class MultiHeadAttention(nn.Module):
    """Multiple heads arranged in parallel for Self-Attention."""

    def __init__(self, num_heads, embedding_dimension, block_size):
        super().__init__()
        head_size = embedding_dimension // num_heads
        self.heads = nn.ModuleList(
            [Head(head_size, embedding_dimension, block_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(embedding_dimension, embedding_dimension)

    def forward(self, x, kv_cache_enabled=False):
        out = torch.cat(
            [head(x, kv_cache_enabled=kv_cache_enabled) for head in self.heads], dim=-1
        )  # B, T, embedding_dimension
        out = self.proj(out)  # B, T, embedding_dimension
        return out

    def reset_kv_cache(self):
        for head in self.heads:
            head.reset_kv_cache()

class FeedForward(nn.Module):
    """Feed Forward Layer"""

    def __init__(self, in_dim):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(in_dim, 4 * in_dim), nn.GELU(), nn.Linear(4 * in_dim, in_dim)
        )

    def forward(self, x):
        return self.feed_forward(x)


class Block(nn.Module):
    """A block of Attention"""

    def __init__(self, embedding_dimension, num_heads, block_size):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            num_heads, embedding_dimension, block_size
        )
        self.ff_layer = FeedForward(embedding_dimension)
        self.ln1 = nn.RMSNorm(embedding_dimension)
        self.ln2 = nn.RMSNorm(embedding_dimension)

    def forward(self, x, kv_cache_enabled=False):
        # Residual Connections
        x = x + self.multi_head_attention(
            self.ln1(x),
            kv_cache_enabled=kv_cache_enabled
        )
        x = x + self.ff_layer(self.ln2(x))
        return x
    
    def reset_kv_cache(self):
        self.multi_head_attention.reset_kv_cache()


class Transformer(nn.Module):
    """Transformer block"""

    def __init__(
        self, vocab_size, embedding_dimension, num_heads, block_size, num_layers
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dimension)
        self.positional_embedding_table = nn.Embedding(block_size, embedding_dimension)
        self.blocks = nn.ModuleList(
            [
                Block(embedding_dimension, num_heads, block_size)
                for _ in range(num_layers)
            ]
        )
        self.ln1 = nn.RMSNorm(embedding_dimension)
        self.linear = nn.Linear(embedding_dimension, vocab_size)
        self.block_size = block_size
        self.current_pos = 0

    def forward(self, x, targets=None, kv_cache_enabled=False):
        B, T = x.shape
        token_embedding = self.token_embedding_table(x)  # B, T, C
        
        if kv_cache_enabled:
            positional_embedding = self.positional_embedding_table(
                torch.arange(self.current_pos, self.current_pos + T)
            )
            self.current_pos += T
        else:
            positional_embedding = self.positional_embedding_table(torch.arange(T))  # T, C
    
        x = token_embedding + positional_embedding  # B, T, C

        for block in self.blocks:
            x = block(x, kv_cache_enabled=kv_cache_enabled)

        logits = self.linear(self.ln1(x))  # B, T, vocab_size

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens=1000):
        for _ in range(max_new_tokens):
            idx_trimmed = idx[:, -self.block_size :]  # B, T
            logits, loss = self.forward(idx_trimmed)  # B, T, C
            next_predicted_logit = logits[:, -1, :]
            probs = F.softmax(next_predicted_logit, dim=-1)
            next_predicted_token = torch.multinomial(probs, num_samples=1)

            # Replace below ID with EOS token ID of the tokenizer you are using.
            if next_predicted_token.item() == 50256:
                break

            idx = torch.cat((idx, next_predicted_token), dim=1)  # B, T+1

        return idx

    def generate_with_kv_cache_enabled(self, idx, max_new_tokens=1000):
        def generate_block(generated_idx, tokens_to_generate):
            self.reset_kv_cache()
            tokens = generated_idx.shape[-1]
            generated_idx_trimmed = generated_idx[:, tokens-1:tokens]
            logits, loss = self.forward(generated_idx_trimmed, kv_cache_enabled=True)

            for _ in range(tokens_to_generate-1):
                next_predicted_logit = logits[:, -1, :]  # B, embedding_dimension
                probs = F.softmax(next_predicted_logit, dim=-1)  # B, 1
                next_predicted_character = torch.multinomial(probs, num_samples=1)
                generated_idx_trimmed = torch.cat(
                    (generated_idx_trimmed, next_predicted_character), 
                    dim=1
                )  # B,T+1
                logits, loss = self.forward(next_predicted_character, kv_cache_enabled=True)
            
            return generated_idx_trimmed

        if max_new_tokens < self.block_size:
            # One partial block to execute
            return torch.cat((idx, generate_block(idx, max_new_tokens)), dim=1)

        num_blocks = max_new_tokens // self.block_size

        if max_new_tokens % self.block_size == 0:
            # Equally divided blocks
            for _ in range(num_blocks):
                idx = torch.cat((idx, generate_block(idx, self.block_size)), dim=1)
        else:
            # Last block is partial
            for _ in range(num_blocks-1):
                idx = torch.cat((idx, generate_block(idx, self.block_size)), dim=1)
            
            idx = torch.cat((idx, generate_block(idx, max_new_tokens%self.block_size)), dim=1)
   
        return idx

    def reset_kv_cache(self):
        for block in self.blocks:
            block.reset_kv_cache()
        self.current_pos = 0