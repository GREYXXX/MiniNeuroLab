import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
import requests
from typing import List

from models.model_lm import GPTModel, GPTConfig
import tiktoken
from pytorch_lightning.callbacks import ModelCheckpoint


@dataclass
class TrainingConfig:
    batch_size: int = 2
    num_epochs: int = 10
    num_workers: int = 0
    train_ratio: float = 0.9


class TextDataset(Dataset):
    def __init__(self, text_data: str, tokenizer, context_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text_data)

        for i in range(0, len(token_ids) - context_length, stride):
            input_chunk = token_ids[i : i + context_length]
            target_chunk = token_ids[i + 1 : i + context_length + 1]

            if (
                len(input_chunk) == context_length
                and len(target_chunk) == context_length
            ):
                self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
                self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def download_training_data() -> str:
    urls = [
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "https://www.gutenberg.org/files/74/74-0.txt",
        "https://www.gutenberg.org/files/1342/1342-0.txt",
    ]

    text_data = ""
    for url in urls:
        try:
            print(f"Downloading from {url}...")
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                content = response.text
                if len(content) > 1000:
                    text_data += content + "\n\n"
                    print(f"Successfully downloaded {len(content):,} characters")
                else:
                    print(f"Content too short, skipping")
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            continue

    if len(text_data) < 10000:
        print("Insufficient data downloaded, using fallback text")
        fallback_text = (
            """
        The study of artificial intelligence represents one of humanity's greatest intellectual challenges.
        Machine learning algorithms can identify patterns in vast datasets that would be impossible for humans to detect manually.
        Natural language processing enables computers to understand, interpret, and generate human language in meaningful ways.
        Deep neural networks, inspired by the structure of biological brains, have revolutionized fields from computer vision to speech recognition.
        The transformer architecture, introduced in the paper "Attention Is All You Need," fundamentally changed how we approach sequence modeling tasks.
        Large language models trained on diverse text corpora can generate coherent, contextually appropriate responses across a wide range of topics.
        The attention mechanism allows models to focus on relevant parts of the input sequence when making predictions.
        Training these models requires substantial computational resources and careful optimization of hyperparameters.
        Fine-tuning pre-trained models for specific tasks has become a standard practice in machine learning.
        The ethical implications of artificial intelligence continue to be an important area of research and discussion.
        Reinforcement learning algorithms learn optimal strategies through trial and error interactions with their environment.
        Computer vision systems can now achieve superhuman performance on tasks like image classification and object detection.
        The democratization of AI tools has enabled researchers and practitioners worldwide to build sophisticated applications.
        Transfer learning leverages knowledge gained from one task to improve performance on related tasks.
        The field of AI continues to evolve rapidly, with new breakthroughs emerging regularly.
        """
            * 200
        )
        text_data = fallback_text

    print(f"Total training data: {len(text_data):,} characters")
    return text_data


def create_data_loaders(
    text_data: str, tokenizer, config: GPTConfig, train_config: TrainingConfig
):
    split_idx = int(train_config.train_ratio * len(text_data))

    train_dataset = TextDataset(
        text_data[:split_idx], tokenizer, config.context_length, config.context_length
    )

    val_dataset = TextDataset(
        text_data[split_idx:], tokenizer, config.context_length, config.context_length
    )

    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=train_config.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=train_config.num_workers,
    )

    return train_loader, val_loader


def train_gpt():
    torch.manual_seed(123)

    gpt_config = GPTConfig(
        vocab_size=50257,
        context_length=256,
        emb_dim=384,
        n_heads=6,
        n_layers=6,
        drop_rate=0.1,
        qkv_bias=False,
        learning_rate=1e-3,
        weight_decay=0.1,
    )

    train_config = TrainingConfig(
        batch_size=4, num_epochs=5, num_workers=0, train_ratio=0.9
    )

    print("Downloading and preparing training data...")
    text_data = download_training_data()
    print(f"Total text length: {len(text_data):,} characters")

    print("Creating data loaders...")
    tokenizer = tiktoken.get_encoding("gpt2")
    train_loader, val_loader = create_data_loaders(
        text_data, tokenizer, gpt_config, train_config
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    model = GPTModel(gpt_config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="gpt-epoch{epoch}",
        save_top_k=-1,
        save_last=True,
        every_n_epochs=1,
    )

    trainer = pl.Trainer(
        max_epochs=train_config.num_epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=20,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback],
    )

    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    torch.save(model.state_dict(), "gpt_model.pth")

    return model, trainer, tokenizer


if __name__ == "__main__":
    model, trainer, tokenizer = train_gpt()
