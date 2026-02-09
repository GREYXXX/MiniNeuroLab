"""
Diffusion Scheduler

Implements DDPM (Denoising Diffusion Probabilistic Models) noise scheduling.
Handles forward diffusion (adding noise) and reverse diffusion (denoising steps).
"""

import torch
import torch.nn.functional as F
from typing import Tuple


class DDPMScheduler:
    """
    DDPM noise scheduler for diffusion models.

    Implements the forward and reverse diffusion processes:
    - Forward: q(x_t | x_0) - gradually add noise
    - Reverse: p(x_{t-1} | x_t) - gradually remove noise

    Uses a linear beta schedule by default, but can be extended
    to cosine or other schedules.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: str = "linear",
    ):
        """
        Initialize DDPM scheduler.

        Args:
            num_timesteps: Number of diffusion timesteps
            beta_start: Starting value of beta (noise schedule)
            beta_end: Ending value of beta
            schedule_type: Type of schedule ("linear" or "cosine")
        """
        self.num_timesteps = num_timesteps

        # Compute beta schedule
        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "cosine":
            # Cosine schedule (better for high-resolution images)
            s = 0.008
            steps = torch.arange(num_timesteps + 1, dtype=torch.float32)
            alphas_cumprod = (
                torch.cos(((steps / num_timesteps) + s) / (1 + s) * torch.pi / 2) ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        # Pre-compute values for efficiency
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for forward diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for reverse diffusion q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            torch.sqrt(self.alphas_cumprod_prev)
            * self.betas
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            torch.sqrt(self.alphas)
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )

    def add_noise(
        self, x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward diffusion process: add noise to clean data.

        q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

        Args:
            x0: Clean data [batch_size, ...]
            noise: Random noise [batch_size, ...]
            timesteps: Timestep indices [batch_size]

        Returns:
            xt: Noisy data [batch_size, ...]
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(x0.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(
            x0.device
        )

        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(x0.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        while len(sqrt_one_minus_alpha_prod.shape) < len(x0.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        return sqrt_alpha_prod * x0 + sqrt_one_minus_alpha_prod * noise

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Reverse diffusion step: denoise sample.

        Predicts x_{t-1} from x_t using the model's noise prediction.

        Args:
            model_output: Predicted noise from model [batch_size, ...]
            timestep: Current timestep (integer)
            sample: Current noisy sample x_t [batch_size, ...]
            eta: Stochasticity parameter (0 = DDPM, 1 = DDIM)

        Returns:
            pred_prev_sample: Predicted previous sample x_{t-1} [batch_size, ...]
        """
        t = timestep

        # Get parameters for this timestep
        beta_t = self.betas[t].to(sample.device)
        alpha_t = self.alphas[t].to(sample.device)
        alpha_prod_t = self.alphas_cumprod[t].to(sample.device)
        alpha_prod_t_prev = self.alphas_cumprod_prev[t].to(sample.device)

        # Predict x0 from noise prediction
        pred_original_sample = (
            sample - torch.sqrt(1 - alpha_prod_t) * model_output
        ) / torch.sqrt(alpha_prod_t)

        # Compute mean of posterior q(x_{t-1} | x_t, x_0)
        pred_prev_sample_mean = (
            torch.sqrt(alpha_prod_t_prev) * beta_t * pred_original_sample
            + torch.sqrt(alpha_t) * (1 - alpha_prod_t_prev) * sample
        ) / (1 - alpha_prod_t)

        # Compute variance
        prev_sample_variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * beta_t

        if t > 0:
            # Add noise for stochastic sampling
            noise = torch.randn_like(sample)
            variance = torch.sqrt(prev_sample_variance) * eta * noise
            pred_prev_sample = pred_prev_sample_mean + variance
        else:
            # At t=0, no noise needed
            pred_prev_sample = pred_prev_sample_mean

        return pred_prev_sample

    def scale_input(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Scale input to the range expected by the model.

        Args:
            sample: Input sample

        Returns:
            scaled_sample: Scaled sample
        """
        return sample

    def get_velocity(
        self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Get velocity prediction target (alternative to noise prediction).

        Some models predict velocity instead of noise for better training stability.

        Args:
            sample: Noisy sample x_t
            noise: Ground truth noise
            timesteps: Timestep indices

        Returns:
            velocity: Velocity target
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(sample.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(
            sample.device
        )

        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity
