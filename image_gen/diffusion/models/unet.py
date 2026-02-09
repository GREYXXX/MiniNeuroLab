"""
U-Net Denoising Model

U-Net architecture for predicting noise in diffusion models.
Includes self-attention and cross-attention mechanisms for text conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal position embedding for timesteps.

    Creates embeddings that encode the diffusion timestep,
    allowing the model to condition on the noise level.
    """

    def __init__(self, dim: int):
        """
        Initialize sinusoidal position embedding.

        Args:
            dim: Embedding dimension
        """
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal embeddings for timesteps.

        Args:
            timesteps: Timestep values [batch_size]

        Returns:
            embeddings: Sinusoidal embeddings [batch_size, dim]
        """
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None].float() * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)

        # Pad if dim is odd
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))

        return embeddings


class CrossAttention(nn.Module):
    """
    Cross-attention layer for text-image alignment.

    Allows the model to attend to text embeddings while processing
    image features, enabling text-conditioned image generation.
    """

    def __init__(
        self, query_dim: int, context_dim: int, num_heads: int = 8, dropout: float = 0.0
    ):
        """
        Initialize cross-attention layer.

        Args:
            query_dim: Dimension of query (image features)
            context_dim: Dimension of context (text embeddings)
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim**-0.5

        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"

        # Query, Key, Value projections
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)
        self.to_v = nn.Linear(context_dim, query_dim)
        self.to_out = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-attention.

        Args:
            x: Image features [batch_size, N, query_dim]
            context: Text features [batch_size, M, context_dim]

        Returns:
            out: Attended features [batch_size, N, query_dim]
        """
        B, N, _ = x.shape

        # Compute Q, K, V
        q = self.to_q(x)  # [B, N, query_dim]
        k = self.to_k(context)  # [B, M, query_dim]
        v = self.to_v(context)  # [B, M, query_dim]

        # Reshape for multi-head attention
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [B, heads, N, head_dim]
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [B, heads, M, head_dim]
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [B, heads, M, head_dim]

        # Attention computation
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, heads, N, M]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, heads, N, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, N, -1)  # [B, N, query_dim]

        return self.to_out(out)


class UNetBlock(nn.Module):
    """
    U-Net block with self-attention and cross-attention.

    Each block processes features at a specific resolution level,
    incorporating time embeddings and text conditioning.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        text_emb_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        """
        Initialize U-Net block.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            time_emb_dim: Time embedding dimension
            text_emb_dim: Text embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Normalization layers
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            out_channels, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_sa = nn.GroupNorm(32, out_channels)

        # Cross-attention with text
        self.cross_attn = CrossAttention(out_channels, text_emb_dim, num_heads, dropout)
        self.norm_ca = nn.GroupNorm(32, out_channels)

        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(
        self, x: torch.Tensor, time_emb: torch.Tensor, text_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through U-Net block.

        Args:
            x: Input features [batch_size, in_channels, H, W]
            time_emb: Time embedding [batch_size, time_emb_dim]
            text_emb: Text embedding [batch_size, seq_len, text_emb_dim]

        Returns:
            out: Output features [batch_size, out_channels, H, W]
        """
        residual = self.residual_conv(x)

        # First convolution + time embedding
        h = self.conv1(x)
        h = self.norm1(h)
        h = h + self.time_mlp(time_emb)[:, :, None, None]
        h = F.silu(h)

        # Second convolution
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        # Self-attention
        B, C, H, W = h.shape
        h_flat = h.view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]
        h_attn, _ = self.self_attn(h_flat, h_flat, h_flat)
        h_attn = h_attn.transpose(1, 2).view(B, C, H, W)
        h = h + h_attn
        h = self.norm_sa(h)

        # Cross-attention with text
        h_flat = h.view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]
        h_cross = self.cross_attn(h_flat, text_emb)
        h_cross = h_cross.transpose(1, 2).view(B, C, H, W)
        h = h + h_cross
        h = self.norm_ca(h)

        return h + residual


class DenoisingUNet(nn.Module):
    """
    U-Net for denoising diffusion with text conditioning.

    Architecture:
    - Encoder: Downsampling blocks with attention
    - Bottleneck: Highest level features
    - Decoder: Upsampling blocks with skip connections
    - Time and text conditioning throughout

    This is the standard architecture used in Stable Diffusion.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        text_emb_dim: int = 512,
        base_channels: int = 128,
        num_res_blocks: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        """
        Initialize denoising U-Net.

        Args:
            in_channels: Input channels (latent space channels)
            out_channels: Output channels (predicted noise channels)
            text_emb_dim: Text embedding dimension
            base_channels: Base number of channels
            num_res_blocks: Number of residual blocks per resolution
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        # Time embedding
        time_emb_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Encoder blocks (downsampling)
        self.enc1 = UNetBlock(
            in_channels, base_channels, time_emb_dim, text_emb_dim, num_heads, dropout
        )
        self.enc2 = UNetBlock(
            base_channels,
            base_channels * 2,
            time_emb_dim,
            text_emb_dim,
            num_heads,
            dropout,
        )
        self.enc3 = UNetBlock(
            base_channels * 2,
            base_channels * 4,
            time_emb_dim,
            text_emb_dim,
            num_heads,
            dropout,
        )

        # Bottleneck
        self.bottleneck = UNetBlock(
            base_channels * 4,
            base_channels * 4,
            time_emb_dim,
            text_emb_dim,
            num_heads,
            dropout,
        )

        # Decoder blocks (upsampling with skip connections)
        self.dec3 = UNetBlock(
            base_channels * 8,
            base_channels * 2,
            time_emb_dim,
            text_emb_dim,
            num_heads,
            dropout,
        )
        self.dec2 = UNetBlock(
            base_channels * 4,
            base_channels,
            time_emb_dim,
            text_emb_dim,
            num_heads,
            dropout,
        )
        self.dec1 = UNetBlock(
            base_channels * 2,
            base_channels,
            time_emb_dim,
            text_emb_dim,
            num_heads,
            dropout,
        )

        # Output layer
        self.out = nn.Conv2d(base_channels, out_channels, 1)

        # Downsampling and upsampling
        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor, text_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through U-Net.

        Args:
            x: Noisy latent [batch_size, in_channels, H, W]
            timesteps: Diffusion timesteps [batch_size]
            text_emb: Text conditioning [batch_size, seq_len, text_emb_dim]

        Returns:
            noise_pred: Predicted noise [batch_size, out_channels, H, W]
        """
        # Get time embedding
        t_emb = self.time_mlp(timesteps)

        # Encoder with skip connections
        h1 = self.enc1(x, t_emb, text_emb)
        h2 = self.enc2(self.down(h1), t_emb, text_emb)
        h3 = self.enc3(self.down(h2), t_emb, text_emb)

        # Bottleneck
        h = self.bottleneck(self.down(h3), t_emb, text_emb)

        # Decoder with skip connections
        h = self.dec3(torch.cat([self.up(h), h3], dim=1), t_emb, text_emb)
        h = self.dec2(torch.cat([self.up(h), h2], dim=1), t_emb, text_emb)
        h = self.dec1(torch.cat([self.up(h), h1], dim=1), t_emb, text_emb)

        # Output
        return self.out(h)
