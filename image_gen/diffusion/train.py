"""
PyTorch Lightning Training Module

Provides LightningModule for training diffusion models with PyTorch Lightning.
This handles training loop, validation, logging, and checkpointing.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Optional, Dict, Any
import torch.nn.functional as F

from .models import LatentDiffusionModel
from .scheduler import DDPMScheduler
from .data import TextImageDataset, create_data_loaders


class DiffusionLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training diffusion models.

    Handles:
    - Training step with diffusion loss
    - Validation step
    - Optimizer configuration
    - Learning rate scheduling
    - Logging metrics
    """

    def __init__(
        self,
        model: LatentDiffusionModel,
        scheduler: DDPMScheduler,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        kl_loss_weight: float = 0.00001,
        num_timesteps: int = 1000,
        loss_type: str = "mse",  # "mse" or "l1"
    ):
        """
        Initialize Lightning module.

        Args:
            model: Latent diffusion model
            scheduler: Diffusion scheduler
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            kl_loss_weight: Weight for VAE KL loss
            num_timesteps: Number of diffusion timesteps
            loss_type: Type of loss function ("mse" or "l1")
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "scheduler"])

        self.model = model
        self.scheduler = scheduler
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.kl_loss_weight = kl_loss_weight
        self.num_timesteps = num_timesteps
        self.loss_type = loss_type

        # Loss function
        if loss_type == "mse":
            self.loss_fn = F.mse_loss
        elif loss_type == "l1":
            self.loss_fn = F.l1_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step.

        Args:
            batch: Batch of data containing 'image' and 'text_tokens'
            batch_idx: Batch index

        Returns:
            Loss value
        """
        images = batch["image"]
        text_tokens = batch["text_tokens"]

        # Forward pass
        outputs = self.model(
            images,
            text_tokens,
            scheduler=self.scheduler,
            num_timesteps=self.num_timesteps,
        )

        # Get losses
        diffusion_loss = outputs["diffusion_loss"]
        kl_loss = outputs["kl_loss"]
        total_loss = outputs["total_loss"]

        # Log metrics
        self.log("train/diffusion_loss", diffusion_loss, on_step=True, on_epoch=True)
        self.log("train/kl_loss", kl_loss, on_step=True, on_epoch=True)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"], on_step=True)

        return total_loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, float]:
        """
        Validation step.

        Args:
            batch: Batch of data
            batch_idx: Batch index

        Returns:
            Dictionary of validation metrics
        """
        images = batch["image"]
        text_tokens = batch["text_tokens"]

        # Forward pass
        outputs = self.model(
            images,
            text_tokens,
            scheduler=self.scheduler,
            num_timesteps=self.num_timesteps,
        )

        # Get losses
        diffusion_loss = outputs["diffusion_loss"]
        kl_loss = outputs["kl_loss"]
        total_loss = outputs["total_loss"]

        # Log metrics
        self.log("val/diffusion_loss", diffusion_loss, on_step=False, on_epoch=True)
        self.log("val/kl_loss", kl_loss, on_step=False, on_epoch=True)
        self.log("val/total_loss", total_loss, on_step=False, on_epoch=True)

        return {
            "val_loss": total_loss.item(),
            "val_diffusion_loss": diffusion_loss.item(),
            "val_kl_loss": kl_loss.item(),
        }

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

        Returns:
            Optimizer and scheduler configuration
        """
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )

        # Learning rate scheduler (cosine annealing)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.learning_rate * 0.01,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        # Log learning rate
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("train/epoch_lr", lr, on_epoch=True)


def train_diffusion_model(
    model: LatentDiffusionModel,
    scheduler: DDPMScheduler,
    train_dataset: TextImageDataset,
    val_dataset: Optional[TextImageDataset] = None,
    config: Optional[Dict[str, Any]] = None,
) -> pl.Trainer:
    """
    Train diffusion model using PyTorch Lightning.

    Args:
        model: Latent diffusion model
        scheduler: Diffusion scheduler
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        config: Training configuration dictionary

    Returns:
        Trained trainer object
    """
    # Default configuration
    default_config = {
        "batch_size": 4,
        "num_workers": 0,
        "max_epochs": 10,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "kl_loss_weight": 0.00001,
        "num_timesteps": 1000,
        "loss_type": "mse",
        "accelerator": "auto",
        "devices": 1,
        "precision": "16-mixed",  # Use mixed precision for faster training
        "gradient_clip_val": 1.0,
        "checkpoint_dir": "./checkpoints",
        "log_dir": "./logs",
    }

    if config:
        default_config.update(config)

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        batch_size=default_config["batch_size"],
        num_workers=default_config["num_workers"],
    )

    # Create Lightning module
    lightning_module = DiffusionLightningModule(
        model=model,
        scheduler=scheduler,
        learning_rate=default_config["learning_rate"],
        weight_decay=default_config["weight_decay"],
        kl_loss_weight=default_config["kl_loss_weight"],
        num_timesteps=default_config["num_timesteps"],
        loss_type=default_config["loss_type"],
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=default_config["checkpoint_dir"],
        filename="diffusion-{epoch:02d}-{val_loss:.2f}",
        monitor="val/total_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Logger
    logger = TensorBoardLogger(
        save_dir=default_config["log_dir"],
        name="diffusion",
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=default_config["max_epochs"],
        accelerator=default_config["accelerator"],
        devices=default_config["devices"],
        precision=default_config["precision"],
        gradient_clip_val=default_config["gradient_clip_val"],
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=0.5,  # Validate twice per epoch
    )

    # Train
    trainer.fit(lightning_module, train_loader, val_loader)

    return trainer
