import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
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
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert (
            config.emb_dim % config.n_heads == 0
        ), "emb_dim must be divisible by n_heads"

        self.emb_dim = config.emb_dim
        self.n_heads = config.n_heads
        self.head_dim = config.emb_dim // config.n_heads

        self.q_proj = nn.Linear(config.emb_dim, config.emb_dim, bias=config.qkv_bias)
        self.k_proj = nn.Linear(config.emb_dim, config.emb_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(config.emb_dim, config.emb_dim, bias=config.qkv_bias)
        self.out_proj = nn.Linear(config.emb_dim, config.emb_dim)
        self.dropout = nn.Dropout(config.drop_rate)

        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(config.context_length, config.context_length), diagonal=1
            ),
        )

    def forward(self, x):
        batch_size, seq_len, d_emb = x.shape

        k_states = self.k_proj(x)
        q_states = self.q_proj(x)
        v_states = self.v_proj(x)

        # Reshape and transpose for multi-head attention
        # (B, S, D) -> (B, S, n_head, head_dim) -> (B, n_head, S, head_dim)
        k_states = k_states.view(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)
        v_states = v_states.view(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)
        q_states = q_states.view(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)

        # Dot product attention and apply mask
        # (B, n_head, S, head_dim) @ (B, n_head, head_dim, S) -> (B, n_head, S, S)
        attn_scores = q_states @ k_states.transpose(2, 3)
        mask_bool = self.mask.bool()[:seq_len, :seq_len]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Softmax attention weights
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted aggregation of values
        # (B, n_head, S, S) @ (B, n_head, S, head_dim) -> (B, n_head, S, head_dim)
        context_vec = (attn_weights @ v_states).transpose(1, 2)

        # Reshape and output projection
        # (B, n_head, S, head_dim) -> (B, S, n_head * head_dim)
        context_vec = context_vec.reshape(batch_size, seq_len, self.emb_dim)
        context_vec = self.out_proj(context_vec)

        return context_vec


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.emb_dim, 4 * config.emb_dim),
            GELU(),
            nn.Linear(4 * config.emb_dim, config.emb_dim),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.att = MultiHeadAttention(config)
        self.ff = FeedForward(config)
        self.norm1 = LayerNorm(config.emb_dim)
        self.norm2 = LayerNorm(config.emb_dim)
        self.drop_shortcut = nn.Dropout(config.drop_rate)

    def forward(self, x):
        # Skip connection for attention
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Skip connection for feed forward
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPTModel(pl.LightningModule):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_emb = nn.Embedding(config.context_length, config.emb_dim)
        self.drop_emb = nn.Dropout(config.drop_rate)

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.final_norm = LayerNorm(config.emb_dim)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    def training_step(self, batch, batch_idx):
        input_batch, target_batch = batch
        logits = self(input_batch)
        loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        preds = torch.argmax(logits, dim=-1)
        correct = (preds == target_batch).float().mean()
        perplexity = torch.exp(loss.detach())
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_ppl", perplexity, prog_bar=True, on_epoch=True)
        self.log("train_acc", correct, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_batch, target_batch = batch
        logits = self(input_batch)
        loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        preds = torch.argmax(logits, dim=-1)
        correct = (preds == target_batch).float().mean()
        perplexity = torch.exp(loss.detach())
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_ppl", perplexity, prog_bar=True, on_epoch=True)
        self.log("val_acc", correct, prog_bar=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if "val_loss" in metrics:
            print(
                f"\n[Validation] Epoch {self.current_epoch} | "
                f"Loss: {metrics['val_loss']:.4f} | "
                f"PPL: {metrics['val_ppl']:.4f} | "
                f"Accuracy: {metrics['val_acc']:.4f}"
            )

    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if "train_loss" in metrics:
            print(
                f"\n[Train] Epoch {self.current_epoch} | "
                f"Loss: {metrics['train_loss']:.4f} | "
                f"PPL: {metrics['train_ppl']:.4f} | "
                f"Accuracy: {metrics['train_acc']:.4f}"
            )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def generate_text(
        self, tokenizer, start_context, max_new_tokens=50, temperature=1.0
    ):
        self.eval()
        context_size = self.pos_emb.weight.shape[0]

        encoded = tokenizer.encode(start_context)
        encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(self.device)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = encoded_tensor[:, -context_size:]
                logits = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, 1)
                encoded_tensor = torch.cat([encoded_tensor, idx_next], dim=1)

        decoded_text = tokenizer.decode(encoded_tensor.squeeze(0).tolist())
        self.train()
        return decoded_text
