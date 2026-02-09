"""
Diffusion Transformer (DiT) Module

Transformer-based architecture for diffusion denoising.
This is an alternative to U-Net that uses Vision Transformers (ViT)
for processing image latents in a patch-based manner.

Reference: "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None].float() * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)

        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))

        return embeddings


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization.

    Conditioned on timestep and text embeddings, allowing the model
    to adapt its normalization based on the diffusion timestep.
    """

    def __init__(self, hidden_size: int, condition_dim: int):
        """
        Initialize AdaLN.

        Args:
            hidden_size: Hidden dimension
            condition_dim: Condition embedding dimension
        """
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ada_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, hidden_size * 2),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive layer normalization.

        Args:
            x: Input features [batch_size, seq_len, hidden_size]
            condition: Condition embedding [batch_size, condition_dim]

        Returns:
            out: Normalized features [batch_size, seq_len, hidden_size]
        """
        shift, scale = self.ada_mlp(condition).chunk(2, dim=-1)
        x = self.norm(x)
        x = x * (1 + scale[:, None, :]) + shift[:, None, :]
        return x


class DiTBlock(nn.Module):
    """
    Diffusion Transformer block.

    Combines self-attention, cross-attention (for text), and feed-forward
    networks with adaptive layer normalization.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        text_emb_dim: int = 512,
        condition_dim: int = 512,
        dropout: float = 0.0,
    ):
        """
        Initialize DiT block.

        Args:
            hidden_size: Hidden dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio for MLP hidden dimension
            text_emb_dim: Text embedding dimension
            condition_dim: Condition embedding dimension (time + text pooled)
            dropout: Dropout rate
        """
        super().__init__()

        # Adaptive layer norms
        self.ada_ln1 = AdaLN(hidden_size, condition_dim)
        self.ada_ln2 = AdaLN(hidden_size, condition_dim)
        self.ada_ln3 = AdaLN(hidden_size, condition_dim)

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention for text conditioning
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward network
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout),
        )

        # Text projection for cross-attention
        self.text_proj = nn.Linear(text_emb_dim, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through DiT block.

        Args:
            x: Input features [batch_size, seq_len, hidden_size]
            condition: Condition embedding [batch_size, condition_dim]
            text_emb: Text embeddings [batch_size, text_seq_len, text_emb_dim]

        Returns:
            out: Output features [batch_size, seq_len, hidden_size]
        """
        # Self-attention
        x_norm = self.ada_ln1(x, condition)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Cross-attention with text (if provided)
        if text_emb is not None:
            x_norm = self.ada_ln2(x, condition)
            text_proj = self.text_proj(text_emb)
            cross_out, _ = self.cross_attn(x_norm, text_proj, text_proj)
            x = x + cross_out

        # Feed-forward network
        x_norm = self.ada_ln3(x, condition)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out

        return x


class PatchEmbed(nn.Module):
    """Patch embedding layer for converting image latents to patches."""

    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ):
        """
        Initialize patch embedding.

        Args:
            img_size: Image size (H, W)
            patch_size: Patch size
            in_channels: Input channels
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert image to patches.

        Args:
            x: Image latents [batch_size, in_channels, H, W]

        Returns:
            patches: Patch embeddings [batch_size, num_patches, embed_dim]
        """
        x = self.proj(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class DiffusionTransformer(nn.Module):
    """
    Diffusion Transformer for denoising.

    Architecture:
    - Patch embedding: Convert image latents to patches
    - Positional embeddings: Learnable position embeddings
    - DiT blocks: Transformer blocks with adaptive layer norm
    - Output projection: Convert patches back to image latents

    This architecture processes images as sequences of patches,
    similar to Vision Transformers, but adapted for diffusion.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (32, 32),
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        text_emb_dim: int = 512,
        time_emb_dim: int = 256,
        dropout: float = 0.0,
    ):
        """
        Initialize Diffusion Transformer.

        Args:
            img_size: Image size in latent space (H, W)
            patch_size: Patch size for patch embedding
            in_channels: Input channels (latent space channels)
            hidden_size: Hidden dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: Ratio for MLP hidden dimension
            text_emb_dim: Text embedding dimension
            time_emb_dim: Time embedding dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, hidden_size)
        num_patches = self.patch_embed.num_patches

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, hidden_size) * 0.02)

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Condition dimension (time + text pooled)
        condition_dim = hidden_size + text_emb_dim  # Time embedding + pooled text

        # DiT blocks
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    text_emb_dim=text_emb_dim,
                    condition_dim=condition_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_size)

        # Output projection (back to image latents)
        self.output_proj = nn.Linear(hidden_size, in_channels * patch_size * patch_size)
        self.patch_size_out = patch_size

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor, text_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through Diffusion Transformer.

        Args:
            x: Noisy latent [batch_size, in_channels, H, W]
            timesteps: Diffusion timesteps [batch_size]
            text_emb: Text conditioning [batch_size, seq_len, text_emb_dim]

        Returns:
            noise_pred: Predicted noise [batch_size, in_channels, H, W]
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, hidden_size]

        # Add positional embedding
        x = x + self.pos_embed

        # Time embedding
        t_emb = self.time_embed(timesteps)  # [B, hidden_size]

        # Pool text embeddings (mean pooling)
        text_pooled = text_emb.mean(dim=1)  # [B, text_emb_dim]

        # Combine time and text for condition
        condition = torch.cat([t_emb, text_pooled], dim=-1)  # [B, condition_dim]

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, condition, text_emb)

        # Final layer norm
        x = self.final_norm(x)

        # Output projection
        x = self.output_proj(x)  # [B, num_patches, in_channels * patch_size^2]

        # Reshape back to image format
        H, W = self.img_size
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        x = x.reshape(B, patch_h, patch_w, self.patch_size, self.patch_size, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(B, -1, H, W)

        return x
