"""
Latent Diffusion Model

Complete model combining text encoder, VAE, and denoising network (U-Net or DiT).
This is the main model class that orchestrates the entire diffusion process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Literal, Any

from .text_encoder import TextEncoder
from .vae import VAEEncoder, VAEDecoder
from .unet import DenoisingUNet
from .dit import DiffusionTransformer


class LatentDiffusionModel(nn.Module):
    """
    Complete Latent Diffusion Model for text-to-image generation.

    Architecture:
    1. Text Encoder: Converts text prompts to embeddings
    2. VAE Encoder: Compresses images to latent space
    3. Denoising Network: Predicts noise in latent space (U-Net or DiT)
    4. VAE Decoder: Reconstructs images from denoised latents

    This follows the Stable Diffusion architecture but allows
    switching between U-Net and DiT for the denoising network.
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        text_emb_dim: int = 512,
        latent_dim: int = 4,
        denoising_type: Literal["unet", "dit"] = "unet",
        unet_config: Optional[dict] = None,
        dit_config: Optional[dict] = None,
        vae_config: Optional[dict] = None,
    ):
        """
        Initialize Latent Diffusion Model.

        Args:
            vocab_size: Vocabulary size for text encoder
            text_emb_dim: Text embedding dimension
            latent_dim: Latent space dimension
            denoising_type: Type of denoising network ("unet" or "dit")
            unet_config: Configuration for U-Net
            dit_config: Configuration for DiT
            vae_config: Configuration for VAE
        """
        super().__init__()

        self.denoising_type = denoising_type
        self.text_emb_dim = text_emb_dim
        self.latent_dim = latent_dim

        # Text encoder
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=text_emb_dim,
        )

        # VAE encoder and decoder
        vae_defaults = {"base_channels": 128, "num_res_blocks": 2}
        if vae_config:
            vae_defaults.update(vae_config)

        self.vae_encoder = VAEEncoder(
            in_channels=3,
            latent_dim=latent_dim,
            **vae_defaults,
        )
        self.vae_decoder = VAEDecoder(
            latent_dim=latent_dim,
            out_channels=3,
            **vae_defaults,
        )

        # Denoising network (U-Net or DiT)
        if denoising_type == "unet":
            unet_defaults = {
                "in_channels": latent_dim,
                "out_channels": latent_dim,
                "text_emb_dim": text_emb_dim,
                "base_channels": 128,
            }
            if unet_config:
                unet_defaults.update(unet_config)
            self.denoising_net = DenoisingUNet(**unet_defaults)
        elif denoising_type == "dit":
            dit_defaults = {
                "img_size": (32, 32),  # Latent space size
                "patch_size": 2,
                "in_channels": latent_dim,
                "hidden_size": 768,
                "depth": 12,
                "num_heads": 12,
                "text_emb_dim": text_emb_dim,
            }
            if dit_config:
                dit_defaults.update(dit_config)
            self.denoising_net = DiffusionTransformer(**dit_defaults)
        else:
            raise ValueError(f"Unknown denoising type: {denoising_type}")

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Encode text to embeddings.

        Args:
            text_tokens: Tokenized text [batch_size, seq_len]

        Returns:
            text_embeddings: Text embeddings [batch_size, seq_len, text_emb_dim]
        """
        return self.text_encoder(text_tokens)

    def encode_image(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode images to latent space.

        Args:
            images: Input images [batch_size, 3, H, W]

        Returns:
            z: Latent representation [batch_size, latent_dim, H//8, W//8]
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        return self.vae_encoder(images)

    def decode_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to images.

        Args:
            latents: Latent representation [batch_size, latent_dim, H//8, W//8]

        Returns:
            images: Reconstructed images [batch_size, 3, H, W]
        """
        return self.vae_decoder(latents)

    def forward(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        scheduler: Optional[Any] = None,
        num_timesteps: int = 1000,
    ) -> dict:
        """
        Training forward pass.

        Args:
            images: Input images [batch_size, 3, H, W]
            text_tokens: Tokenized text [batch_size, seq_len]
            timesteps: Diffusion timesteps [batch_size] (optional, random if None)
            noise: Random noise [batch_size, latent_dim, H//8, W//8] (optional)
            num_timesteps: Number of diffusion timesteps

        Returns:
            Dictionary containing:
                - diffusion_loss: MSE loss for noise prediction
                - kl_loss: VAE KL divergence loss
                - recon_loss: VAE reconstruction loss
                - total_loss: Combined loss
        """
        batch_size = images.shape[0]
        device = images.device

        # 1. Encode text
        text_emb = self.encode_text(text_tokens)

        # 2. Encode images to latent space
        latents, mu, logvar = self.encode_image(images)

        # 3. Sample random timesteps
        if timesteps is None:
            timesteps = torch.randint(
                0, num_timesteps, (batch_size,), device=device, dtype=torch.long
            )

        # 4. Sample noise
        if noise is None:
            noise = torch.randn_like(latents)

        # 5. Add noise to latents using scheduler if provided, otherwise manual
        if scheduler is not None:
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        else:
            # Manual computation if scheduler not provided (for backward compatibility)
            sqrt_alphas_cumprod = torch.sqrt(
                torch.linspace(1.0, 0.0, num_timesteps + 1)[:-1]
            ).to(device)
            sqrt_one_minus_alphas_cumprod = torch.sqrt(
                1.0 - torch.linspace(1.0, 0.0, num_timesteps + 1)[:-1]
            ).to(device)

            alpha_t = sqrt_alphas_cumprod[timesteps]
            beta_t = sqrt_one_minus_alphas_cumprod[timesteps]

            while len(alpha_t.shape) < len(latents.shape):
                alpha_t = alpha_t.unsqueeze(-1)
            while len(beta_t.shape) < len(latents.shape):
                beta_t = beta_t.unsqueeze(-1)

            noisy_latents = alpha_t * latents + beta_t * noise

        # 6. Predict noise with denoising network
        noise_pred = self.denoising_net(noisy_latents, timesteps, text_emb)

        # 7. Compute losses
        # Diffusion loss (main training objective)
        diffusion_loss = F.mse_loss(noise_pred, noise)

        # VAE KL divergence loss (regularization)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

        # VAE reconstruction loss (optional, for VAE training)
        with torch.no_grad():
            recon_images = self.decode_latent(latents)
            recon_loss = F.mse_loss(recon_images, images)

        # Combined loss
        total_loss = diffusion_loss + 0.00001 * kl_loss

        return {
            "diffusion_loss": diffusion_loss,
            "kl_loss": kl_loss,
            "recon_loss": recon_loss,
            "total_loss": total_loss,
        }

    @torch.no_grad()
    def generate(
        self,
        text_tokens: torch.Tensor,
        scheduler,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        image_size: Tuple[int, int] = (256, 256),
        latent_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Generate images from text using diffusion process.

        Args:
            text_tokens: Tokenized text [batch_size, seq_len]
            scheduler: Diffusion scheduler
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            image_size: Output image size (H, W)
            latent_size: Latent space size (H//8, W//8), inferred if None

        Returns:
            generated_images: Generated images [batch_size, 3, H, W]
        """
        device = text_tokens.device
        batch_size = text_tokens.shape[0]

        # Infer latent size from image size
        if latent_size is None:
            latent_size = (image_size[0] // 8, image_size[1] // 8)

        # 1. Encode text (conditional)
        text_emb = self.encode_text(text_tokens)

        # 2. For classifier-free guidance, also need unconditional embedding
        uncond_tokens = torch.zeros_like(text_tokens)
        uncond_emb = self.encode_text(uncond_tokens)

        # 3. Start from random noise
        latents = torch.randn(
            batch_size, self.latent_dim, latent_size[0], latent_size[1], device=device
        )

        # 4. Denoising loop
        timesteps = torch.linspace(
            scheduler.num_timesteps - 1, 0, num_inference_steps
        ).long()

        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict noise with conditional text
            noise_pred_cond = self.denoising_net(latents, t_batch, text_emb)

            # Predict noise with unconditional text
            noise_pred_uncond = self.denoising_net(latents, t_batch, uncond_emb)

            # Apply classifier-free guidance
            # noise_pred = uncond + guidance_scale * (cond - uncond)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # Denoise step using scheduler
            latents = scheduler.step(noise_pred, t.item(), latents)

        # 5. Decode latents to images
        images = self.decode_latent(latents)

        # Clamp to valid range
        images = torch.clamp(images, -1, 1)

        return images
