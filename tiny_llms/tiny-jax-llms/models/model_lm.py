import jax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 256
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = False
    learning_rate: float = 5e-4
    weight_decay: float = 0.1


class LayerNorm(nn.Module):
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        scale = self.param('scale', nn.initializers.ones, x.shape[-1:])
        shift = self.param('shift', nn.initializers.zeros, x.shape[-1:])
        norm_x = (x - mean) / jnp.sqrt(var + self.eps)
        return scale * norm_x + shift


class GELU(nn.Module):
    @nn.compact
    def __call__(self, x):
        return 0.5 * x * (1 + jnp.tanh(
            jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3))
        ))


class MultiHeadAttention(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x, deterministic=True):
        assert (
            self.config.emb_dim % self.config.n_heads == 0
        ), "emb_dim must be divisible by n_heads"
        
        head_dim = self.config.emb_dim // self.config.n_heads
        batch_size, seq_len, d_emb = x.shape
        
        q = nn.Dense(self.config.emb_dim, use_bias=self.config.qkv_bias, name='q_proj')(x)
        k = nn.Dense(self.config.emb_dim, use_bias=self.config.qkv_bias, name='k_proj')(x)
        v = nn.Dense(self.config.emb_dim, use_bias=self.config.qkv_bias, name='v_proj')(x)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.config.n_heads, head_dim)
        k = k.reshape(batch_size, seq_len, self.config.n_heads, head_dim)
        v = v.reshape(batch_size, seq_len, self.config.n_heads, head_dim)
        
        # Transpose to (batch, n_heads, seq_len, head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        # Attention scores
        attn_scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2)))
        
        # Apply causal mask
        mask = jnp.triu(jnp.ones((seq_len, seq_len)), k=1)
        attn_scores = jnp.where(mask.astype(bool), -jnp.inf, attn_scores)
        
        # Scale and softmax
        attn_weights = jax.nn.softmax(attn_scores / jnp.sqrt(head_dim), axis=-1)
        
        # Apply dropout during training
        if not deterministic:
            attn_weights = nn.Dropout(self.config.drop_rate)(attn_weights, deterministic=deterministic)
        
        # Apply attention to values
        context_vec = jnp.matmul(attn_weights, v)
        
        # Reshape back
        context_vec = jnp.transpose(context_vec, (0, 2, 1, 3))
        context_vec = context_vec.reshape(batch_size, seq_len, d_emb)
        
        return nn.Dense(self.config.emb_dim, name='out_proj')(context_vec)


class FeedForward(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(4 * self.config.emb_dim)(x)
        x = GELU()(x)
        x = nn.Dense(self.config.emb_dim)(x)
        return x


class TransformerBlock(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x, deterministic=True):
        # Attention with residual connection
        attn_out = MultiHeadAttention(self.config)(LayerNorm()(x), deterministic=deterministic)
        if not deterministic:
            attn_out = nn.Dropout(self.config.drop_rate)(attn_out, deterministic=deterministic)
        x = x + attn_out
        
        # Feed forward with residual connection
        ff_out = FeedForward(self.config)(LayerNorm()(x))
        if not deterministic:
            ff_out = nn.Dropout(self.config.drop_rate)(ff_out, deterministic=deterministic)
        x = x + ff_out
        
        return x


class GPTModel(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, input_ids, deterministic=True):
        batch_size, seq_len = input_ids.shape
        
        tok_embed = nn.Embed(self.config.vocab_size, self.config.emb_dim)
        pos_embed = nn.Embed(self.config.context_length, self.config.emb_dim)
        
        tok_embeds = tok_embed(input_ids)
        pos_embeds = pos_embed(jnp.arange(seq_len))
        
        x = tok_embeds + pos_embeds
        
        if not deterministic:
            x = nn.Dropout(self.config.drop_rate)(x, deterministic=deterministic)
        
        # Transformer blocks
        for _ in range(self.config.n_layers):
            x = TransformerBlock(self.config)(x, deterministic=deterministic)
        
        # Final layer norm and output projection
        x = LayerNorm()(x)
        logits = nn.Dense(self.config.vocab_size, use_bias=False)(x)
        
        return logits