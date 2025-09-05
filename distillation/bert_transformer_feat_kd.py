import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import numpy as np


class SmallTransformerStudent(nn.Module):
    def __init__(
        self, embed_dim=128, num_heads=2, num_layers=2, hidden_dim=256, num_classes=2
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embedding = nn.Embedding(30522, embed_dim)
        self.feature_proj = nn.Linear(embed_dim, 768)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        hidden = self.encoder(x)
        pooled = hidden.mean(dim=1)

        features = self.feature_proj(pooled)
        logits = self.classifier(self.dropout(pooled))

        return logits, features


class BERTTeacher(nn.Module):
    def __init__(self, pretrained_model="bert-base-uncased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = AutoModel.from_pretrained(pretrained_model)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state.mean(dim=1)
            return pooled_output


class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class FeatureDistillationModule(pl.LightningModule):
    def __init__(self, teacher, student, alpha=0.7, beta=0.3, lr=1e-3):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.alpha = alpha
        self.beta = beta
        self.lr = lr

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        teacher_features = self.teacher(batch["input_ids"], batch["attention_mask"])
        student_logits, student_features = self.student(batch["input_ids"])
        task_loss = self.ce_loss(student_logits, batch["labels"])
        feature_loss = self.mse_loss(student_features, teacher_features)
        total_loss = self.alpha * feature_loss + self.beta * task_loss
        acc = (student_logits.argmax(dim=1) == batch["labels"]).float().mean()

        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        self.log("feature_loss", feature_loss)
        self.log("task_loss", task_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        teacher_features = self.teacher(batch["input_ids"], batch["attention_mask"])
        student_logits, student_features = self.student(batch["input_ids"])

        task_loss = self.ce_loss(student_logits, batch["labels"])
        feature_loss = self.mse_loss(student_features, teacher_features)
        total_loss = self.alpha * feature_loss + self.beta * task_loss

        acc = (student_logits.argmax(dim=1) == batch["labels"]).float().mean()

        self.log("val_loss", total_loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_feature_loss", feature_loss)
        self.log("val_task_loss", task_loss)

        return total_loss

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get("val_loss", 0)
        val_acc = self.trainer.callback_metrics.get("val_acc", 0)
        print(
            f"\nEpoch {self.current_epoch}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.student.parameters(), lr=self.lr)


def load_data(sample_size=2000):
    dataset = load_dataset("imdb")

    if sample_size:
        train_indices = np.random.choice(
            len(dataset["train"]), sample_size, replace=False
        )
        test_indices = np.random.choice(
            len(dataset["test"]), sample_size // 5, replace=False
        )
        train_data = dataset["train"].select(train_indices)
        test_data = dataset["test"].select(test_indices)
    else:
        train_data, test_data = dataset["train"], dataset["test"]

    return (
        train_data["text"],
        train_data["label"],
        test_data["text"],
        test_data["label"],
    )


def main():
    train_texts, train_labels, test_texts, test_labels = load_data()

    teacher = BERTTeacher()
    student = SmallTransformerStudent()

    train_dataset = IMDBDataset(train_texts, train_labels, teacher.tokenizer)
    test_dataset = IMDBDataset(test_texts, test_labels, teacher.tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    model = FeatureDistillationModule(teacher, student)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints/",
        filename="feature-distill-{epoch:02d}-{val_acc:.3f}",
        save_top_k=-1,
        every_n_epochs=1,
    )

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_loader, test_loader)


if __name__ == "__main__":
    main()
