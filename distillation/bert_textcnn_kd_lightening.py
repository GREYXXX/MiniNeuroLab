import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_filters=100, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel_size=k) for k in [3, 4, 5]]
        )
        self.fc = nn.Linear(num_filters * 3, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        conv_out = [F.relu(conv(x)) for conv in self.convs]
        pooled = [F.max_pool1d(out, out.size(2)).squeeze(2) for out in conv_out]
        features = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(features))


class BERTTeacher(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=num_classes
        )
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, vocab_dict, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.vocab_dict = vocab_dict
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        bert_enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        words = text.lower().split()[: self.max_length]
        textcnn_ids = [self.vocab_dict.get(w, 1) for w in words]
        textcnn_ids += [0] * (self.max_length - len(textcnn_ids))

        return {
            "bert_input_ids": bert_enc["input_ids"].squeeze(),
            "bert_attention_mask": bert_enc["attention_mask"].squeeze(),
            "textcnn_input_ids": torch.tensor(textcnn_ids, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class DistillationModule(pl.LightningModule):
    def __init__(self, teacher, student, temperature=3.0, alpha=0.7, lr=1e-3):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        self.lr = lr

    def distillation_loss(self, student_logits, teacher_logits, labels):
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)

        distill_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (
            self.temperature**2
        )
        task_loss = F.cross_entropy(student_logits, labels)

        return self.alpha * distill_loss + (1 - self.alpha) * task_loss

    def training_step(self, batch, batch_idx):
        teacher_logits = self.teacher(
            batch["bert_input_ids"], batch["bert_attention_mask"]
        )
        student_logits = self.student(batch["textcnn_input_ids"])

        loss = self.distillation_loss(student_logits, teacher_logits, batch["labels"])

        acc = (student_logits.argmax(dim=1) == batch["labels"]).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        teacher_logits = self.teacher(
            batch["bert_input_ids"], batch["bert_attention_mask"]
        )
        student_logits = self.student(batch["textcnn_input_ids"])

        loss = self.distillation_loss(student_logits, teacher_logits, batch["labels"])
        acc = (student_logits.argmax(dim=1) == batch["labels"]).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get("val_loss", 0)
        val_acc = self.trainer.callback_metrics.get("val_acc", 0)
        print(
            f"\nEpoch {self.current_epoch}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.student.parameters(), lr=self.lr)


def build_vocab(texts, max_vocab=10000):
    word_count = {}
    for text in texts:
        for word in str(text).lower().split():
            word_count[word] = word_count.get(word, 0) + 1

    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    vocab = {"<PAD>": 0, "<UNK>": 1}

    for word, _ in sorted_words[: max_vocab - 2]:
        vocab[word] = len(vocab)

    return vocab


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

    vocab = build_vocab(train_data["text"])

    return (
        train_data["text"],
        train_data["label"],
        test_data["text"],
        test_data["label"],
        vocab,
    )


def main():
    train_texts, train_labels, test_texts, test_labels, vocab = load_data()

    teacher = BERTTeacher()
    student = TextCNN(len(vocab))

    train_dataset = TextDataset(train_texts, train_labels, teacher.tokenizer, vocab)
    test_dataset = TextDataset(test_texts, test_labels, teacher.tokenizer, vocab)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = DistillationModule(teacher, student)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints/",
        filename="distillation-{epoch:02d}-{val_acc:.3f}",
        save_top_k=-1,  # Save all checkpoints
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
