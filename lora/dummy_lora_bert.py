import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.optim as optim
import math


class DummyDataset(Dataset):
    def __init__(self, tokenizer):
        self.samples = [
            ("I love this movie", 1),
            ("I hate this movie", 0),
            ("This is awesome", 1),
            ("This is terrible", 0),
        ]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=32,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
        }


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=1.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if r > 0:
            self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
            self.B = nn.Parameter(torch.randn(out_features, r) * 0.01)
        else:
            self.A = self.B = None

    def forward(self, x):
        out = F.linear(x, self.weight)
        if self.r > 0:
            lora_out = F.linear(x, self.A.T)
            lora_out = F.linear(lora_out, self.B)
            out += self.alpha * lora_out
        return out


class BertWithLoRA(nn.Module):
    def __init__(self, r=4, alpha=1.0, num_labels=2):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        hidden_size = self.bert.config.hidden_size

        for param in self.bert.parameters():
            param.requires_grad = False

        self.classifier = LoRALinear(hidden_size, num_labels, r=r, alpha=alpha)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = output.last_hidden_state[:, 0]
        logits = self.classifier(cls)
        return logits


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = DummyDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = BertWithLoRA(r=4, alpha=16.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: Loss = {total_loss:.4f}")


if __name__ == "__main__":
    main()
