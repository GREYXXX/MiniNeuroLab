"""
Text Encoder Module

Converts text tokens to semantic embeddings using Transformer architecture.
This is similar to CLIP's text encoder or T5 encoder used in text-to-image models.
"""

import torch
import torch.nn as nn
from typing import Tuple


class TextEncoder(nn.Module):
    """
    Text encoder that converts text tokens to semantic embeddings.

    Architecture:
    - Token embeddings
    - Positional encodings
    - Transformer encoder layers
    - Final projection layer

    This encoder processes text prompts and produces embeddings that condition
    the diffusion model during image generation.
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 512,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 77,
        dropout: float = 0.1,
    ):
        """
        Initialize text encoder.

        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            hidden_dim: Hidden dimension in feed-forward network
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm architecture
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final projection layer
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        # Initialize token embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)

        # Initialize projection layer
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through text encoder.

        Args:
            text_tokens: Tokenized text input [batch_size, seq_len]

        Returns:
            text_embeddings: Semantic embeddings [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len = text_tokens.shape

        # Get token embeddings
        x = self.token_embedding(text_tokens)  # [B, seq_len, embed_dim]

        # Add positional encoding
        x = x + self.pos_embedding[:, :seq_len, :]

        # Apply transformer encoder
        x = self.transformer(x)

        # Final projection and normalization
        x = self.proj(x)
        x = self.layer_norm(x)

        return x

    def encode_pooled(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Encode text and return pooled representation (mean pooling).

        Args:
            text_tokens: Tokenized text input [batch_size, seq_len]

        Returns:
            pooled_embeddings: Pooled embeddings [batch_size, embed_dim]
        """
        embeddings = self.forward(text_tokens)
        # Mean pooling over sequence dimension
        pooled = embeddings.mean(dim=1)
        return pooled
