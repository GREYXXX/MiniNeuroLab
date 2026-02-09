"""
Main Training Script for Diffusion Models

Entry point for training text-to-image diffusion models.
Supports both U-Net and DiT architectures.

Usage:
    python main.py --denoising_type unet --epochs 10
    python main.py --denoising_type dit --epochs 10
"""

import argparse
import torch
import pytorch_lightning as pl
from pathlib import Path

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models import LatentDiffusionModel
from scheduler import DDPMScheduler
from data import create_dummy_dataset, create_data_loaders
from train import DiffusionLightningModule, train_diffusion_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Diffusion Model")

    # Model architecture
    parser.add_argument(
        "--denoising_type",
        type=str,
        default="unet",
        choices=["unet", "dit"],
        help="Type of denoising network (unet or dit)",
    )

    # Training configuration
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")

    # Model configuration
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument(
        "--text_emb_dim", type=int, default=512, help="Text embedding dimension"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=4, help="Latent space dimension"
    )
    parser.add_argument(
        "--num_timesteps", type=int, default=1000, help="Number of diffusion timesteps"
    )

    # Data configuration
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of training samples"
    )
    parser.add_argument(
        "--image_size", type=int, nargs=2, default=[256, 256], help="Image size (H W)"
    )

    # U-Net specific
    parser.add_argument(
        "--unet_base_channels", type=int, default=128, help="U-Net base channels"
    )

    # DiT specific
    parser.add_argument(
        "--dit_hidden_size", type=int, default=768, help="DiT hidden size"
    )
    parser.add_argument("--dit_depth", type=int, default=12, help="DiT depth")
    parser.add_argument(
        "--dit_num_heads", type=int, default=12, help="DiT number of heads"
    )
    parser.add_argument("--dit_patch_size", type=int, default=2, help="DiT patch size")

    # Training settings
    parser.add_argument(
        "--accelerator", type=str, default="auto", help="PyTorch Lightning accelerator"
    )
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument(
        "--precision", type=str, default="16-mixed", help="Training precision"
    )
    parser.add_argument(
        "--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value"
    )

    # Paths
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument("--log_dir", type=str, default="./logs", help="Log directory")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of data loading workers"
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Set random seed
    pl.seed_everything(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 80)
    print("Diffusion Model Training")
    print("=" * 80)
    print(f"Denoising Type: {args.denoising_type}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Image Size: {args.image_size}")
    print("=" * 80)

    # Create directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # Create scheduler
    scheduler = DDPMScheduler(
        num_timesteps=args.num_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        schedule_type="linear",
    )

    # Configure denoising network
    if args.denoising_type == "unet":
        unet_config = {
            "base_channels": args.unet_base_channels,
        }
        dit_config = None
    else:
        unet_config = None
        dit_config = {
            "img_size": (args.image_size[0] // 8, args.image_size[1] // 8),
            "patch_size": args.dit_patch_size,
            "hidden_size": args.dit_hidden_size,
            "depth": args.dit_depth,
            "num_heads": args.dit_num_heads,
        }

    # Create model
    model = LatentDiffusionModel(
        vocab_size=args.vocab_size,
        text_emb_dim=args.text_emb_dim,
        latent_dim=args.latent_dim,
        denoising_type=args.denoising_type,
        unet_config=unet_config,
        dit_config=dit_config,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print()

    # Create datasets
    print("Creating datasets...")
    train_dataset = create_dummy_dataset(
        num_samples=args.num_samples,
        image_size=tuple(args.image_size),
        vocab_size=args.vocab_size,
    )

    val_dataset = create_dummy_dataset(
        num_samples=args.num_samples // 5,
        image_size=tuple(args.image_size),
        vocab_size=args.vocab_size,
    )

    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print()

    # Training configuration
    config = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_timesteps": args.num_timesteps,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "precision": args.precision,
        "gradient_clip_val": args.gradient_clip_val,
        "checkpoint_dir": args.checkpoint_dir,
        "log_dir": args.log_dir,
    }

    # Train model
    print("Starting training...")
    trainer = train_diffusion_model(
        model=model,
        scheduler=scheduler,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
    )

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")
    print(f"Logs saved at: {args.log_dir}")
    print()

    # Example generation (optional)
    print("Example: Generating images...")
    model.eval()
    with torch.no_grad():
        # Create dummy text tokens
        batch_size = 2
        text_tokens = torch.randint(0, args.vocab_size, (batch_size, 77))

        # Generate images
        generated_images = model.generate(
            text_tokens=text_tokens,
            scheduler=scheduler,
            num_inference_steps=50,
            guidance_scale=7.5,
            image_size=tuple(args.image_size),
        )

        print(f"Generated images shape: {generated_images.shape}")
        print(
            f"Image value range: [{generated_images.min():.2f}, {generated_images.max():.2f}]"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
