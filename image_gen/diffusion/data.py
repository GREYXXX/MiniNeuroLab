"""
Dataset and Data Loading Utilities

Provides dataset classes for training diffusion models with text-image pairs.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List
import numpy as np
from PIL import Image
import os


class TextImageDataset(Dataset):
    """
    Dataset for text-image pairs.

    This is a simple dataset class that can be extended to work with
    real datasets like COCO, LAION, or custom datasets.
    """

    def __init__(
        self,
        images: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        text_tokens: Optional[torch.Tensor] = None,
        image_size: Tuple[int, int] = (256, 256),
        transform: Optional[callable] = None,
    ):
        """
        Initialize dataset.

        Args:
            images: Pre-loaded images tensor [N, 3, H, W] or None for dummy data
            texts: List of text strings or None
            text_tokens: Pre-tokenized text tokens [N, seq_len] or None
            image_size: Target image size (H, W)
            transform: Optional image transform function
        """
        self.image_size = image_size
        self.transform = transform

        if images is not None:
            self.images = images
            self.num_samples = len(images)
        else:
            # Create dummy data for demonstration
            self.num_samples = 100
            self.images = None

        if text_tokens is not None:
            self.text_tokens = text_tokens
        elif texts is not None:
            # Simple tokenization (in practice, use proper tokenizer)
            self.text_tokens = self._simple_tokenize(texts)
        else:
            # Create dummy tokens
            self.text_tokens = torch.randint(0, 10000, (self.num_samples, 77))

    def _simple_tokenize(self, texts: List[str]) -> torch.Tensor:
        """
        Simple tokenization (for demonstration).
        In practice, use a proper tokenizer like CLIP's tokenizer.
        """
        # This is a placeholder - use proper tokenization in real scenarios
        max_len = 77
        tokens = []
        for text in texts:
            # Simple character-based tokenization
            text_tokens = [ord(c) % 10000 for c in text[:max_len]]
            text_tokens = text_tokens + [0] * (max_len - len(text_tokens))
            tokens.append(text_tokens)
        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single data sample.

        Returns:
            Dictionary with:
                - image: Image tensor [3, H, W]
                - text_tokens: Text tokens [seq_len]
        """
        if self.images is not None:
            image = self.images[idx]
        else:
            # Generate dummy image
            image = torch.randn(3, self.image_size[0], self.image_size[1])
            # Normalize to [-1, 1] range
            image = torch.tanh(image)

        # Apply transform if provided
        if self.transform:
            image = self.transform(image)

        text_tokens = self.text_tokens[idx]

        return {
            "image": image,
            "text_tokens": text_tokens,
        }


def create_dummy_dataset(
    num_samples: int = 100,
    image_size: Tuple[int, int] = (256, 256),
    vocab_size: int = 10000,
    seq_len: int = 77,
) -> TextImageDataset:
    """
    Create a dummy dataset for testing and demonstration.

    Args:
        num_samples: Number of samples
        image_size: Image size (H, W)
        vocab_size: Vocabulary size for text tokens
        seq_len: Text sequence length

    Returns:
        TextImageDataset instance
    """
    # Create dummy images
    images = torch.randn(num_samples, 3, image_size[0], image_size[1])
    images = torch.tanh(images)  # Normalize to [-1, 1]

    # Create dummy text tokens
    text_tokens = torch.randint(0, vocab_size, (num_samples, seq_len))

    return TextImageDataset(
        images=images,
        text_tokens=text_tokens,
        image_size=image_size,
    )


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    batch_size: int = 4,
    num_workers: int = 0,
    shuffle: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create data loaders for training and validation.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle training data

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader
