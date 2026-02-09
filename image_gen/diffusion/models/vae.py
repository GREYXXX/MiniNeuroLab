"""
VAE (Variational Autoencoder) Module

Compresses images to latent space and reconstructs them back.
This enables diffusion to work in a lower-dimensional latent space,
making training more efficient.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VAEEncoder(nn.Module):
    """
    VAE Encoder: Compresses images to latent space.

    Architecture:
    - Convolutional downsampling blocks
    - Mean and variance prediction heads
    - Reparameterization trick for sampling

    The encoder reduces spatial dimensions by 8x (e.g., 256x256 -> 32x32)
    while increasing channel dimensions to capture semantic information.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 4,
        base_channels: int = 128,
        num_res_blocks: int = 2,
    ):
        """
        Initialize VAE encoder.

        Args:
            in_channels: Number of input image channels (3 for RGB)
            latent_dim: Dimension of latent space
            base_channels: Base number of channels
            num_res_blocks: Number of residual blocks per resolution
        """
        super().__init__()

        self.latent_dim = latent_dim

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Downsampling blocks: 256x256 -> 128x128 -> 64x64 -> 32x32
        self.down1 = self._make_down_block(base_channels, base_channels, num_res_blocks)
        self.down2 = self._make_down_block(
            base_channels, base_channels * 2, num_res_blocks
        )
        self.down3 = self._make_down_block(
            base_channels * 2, base_channels * 4, num_res_blocks
        )

        # Final residual blocks at 32x32
        self.mid = self._make_res_blocks(base_channels * 4, num_res_blocks)

        # Mean and log variance heads
        self.fc_mu = nn.Conv2d(base_channels * 4, latent_dim, 1)
        self.fc_logvar = nn.Conv2d(base_channels * 4, latent_dim, 1)

    def _make_down_block(
        self, in_channels: int, out_channels: int, num_blocks: int
    ) -> nn.Module:
        """Create a downsampling block."""
        blocks = []

        # First block with downsampling
        blocks.append(nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1))
        blocks.append(nn.GroupNorm(32, out_channels))
        blocks.append(nn.SiLU())

        # Additional residual blocks
        for _ in range(num_blocks - 1):
            blocks.append(self._make_res_block(out_channels))

        return nn.Sequential(*blocks)

    def _make_res_blocks(self, channels: int, num_blocks: int) -> nn.Module:
        """Create multiple residual blocks."""
        blocks = [self._make_res_block(channels) for _ in range(num_blocks)]
        return nn.Sequential(*blocks)

    def _make_res_block(self, channels: int) -> nn.Module:
        """Create a single residual block."""
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(32, channels),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE encoder.

        Args:
            x: Input images [batch_size, in_channels, H, W]

        Returns:
            z: Latent representation [batch_size, latent_dim, H//8, W//8]
            mu: Mean of latent distribution [batch_size, latent_dim, H//8, W//8]
            logvar: Log variance of latent distribution [batch_size, latent_dim, H//8, W//8]
        """
        # Initial convolution
        h = self.conv_in(x)
        h = nn.SiLU()(h)

        # Downsampling blocks
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)

        # Middle residual blocks
        h = self.mid(h)

        # Predict mean and log variance
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Reparameterization trick: z = mu + sigma * epsilon
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, logvar


class VAEDecoder(nn.Module):
    """
    VAE Decoder: Reconstructs images from latent space.

    Architecture:
    - Initial projection
    - Upsampling blocks with transposed convolutions
    - Final output layer

    The decoder upsamples latent representations back to image space
    (e.g., 32x32 -> 256x256).
    """

    def __init__(
        self,
        latent_dim: int = 4,
        out_channels: int = 3,
        base_channels: int = 128,
        num_res_blocks: int = 2,
    ):
        """
        Initialize VAE decoder.

        Args:
            latent_dim: Dimension of latent space
            out_channels: Number of output image channels (3 for RGB)
            base_channels: Base number of channels
            num_res_blocks: Number of residual blocks per resolution
        """
        super().__init__()

        # Initial projection
        self.conv_in = nn.Conv2d(latent_dim, base_channels * 4, 3, padding=1)

        # Middle residual blocks at 32x32
        self.mid = self._make_res_blocks(base_channels * 4, num_res_blocks)

        # Upsampling blocks: 32x32 -> 64x64 -> 128x128 -> 256x256
        self.up1 = self._make_up_block(
            base_channels * 4, base_channels * 2, num_res_blocks
        )
        self.up2 = self._make_up_block(base_channels * 2, base_channels, num_res_blocks)
        self.up3 = self._make_up_block(base_channels, base_channels, num_res_blocks)

        # Final output layer
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def _make_up_block(
        self, in_channels: int, out_channels: int, num_blocks: int
    ) -> nn.Module:
        """Create an upsampling block."""
        blocks = []

        # First block with upsampling
        blocks.append(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        )
        blocks.append(nn.GroupNorm(32, out_channels))
        blocks.append(nn.SiLU())

        # Additional residual blocks
        for _ in range(num_blocks - 1):
            blocks.append(self._make_res_block(out_channels))

        return nn.Sequential(*blocks)

    def _make_res_blocks(self, channels: int, num_blocks: int) -> nn.Module:
        """Create multiple residual blocks."""
        blocks = [self._make_res_block(channels) for _ in range(num_blocks)]
        return nn.Sequential(*blocks)

    def _make_res_block(self, channels: int) -> nn.Module:
        """Create a single residual block."""
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(32, channels),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through VAE decoder.

        Args:
            z: Latent representation [batch_size, latent_dim, H//8, W//8]

        Returns:
            x_recon: Reconstructed images [batch_size, out_channels, H, W]
        """
        # Initial projection
        h = self.conv_in(z)
        h = nn.SiLU()(h)

        # Middle residual blocks
        h = self.mid(h)

        # Upsampling blocks
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)

        # Final output
        x_recon = self.conv_out(h)

        # Apply tanh to ensure output is in [-1, 1] range
        x_recon = torch.tanh(x_recon)

        return x_recon
