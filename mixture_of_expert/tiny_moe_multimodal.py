#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Multimodal MoE Pretraining (Toy Example)
- Image encoder: tiny CNN + GAP
- Text encoder: embedding + mean pooling
- MoE layers: shared experts with top-k gating
- Loss: CLIP-style contrastive (InfoNCE) between image/text embeddings after MoE
This is a minimal, educational script, not optimized for performance.
Tested with PyTorch 2.x (CPU).
"""

import math
import random
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


@dataclass
class Config:
    vocab_size: int = 500
    max_len: int = 16
    img_size: int = 64
    batch_size: int = 32
    epochs: int = 2
    lr: float = 3e-4
    d_model: int = 256
    num_experts: int = 4
    top_k: int = 2
    temperature: float = 0.07
    device: str = "cpu"
    seed: int = 42


cfg = Config()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(cfg.seed)


class ToyImageTextDataset(Dataset):
    """Generates random image tensors and random token ids that are weakly correlated.
    This is only for wiring up the model & training loop.
    """

    def __init__(self, n_samples: int = 8192, cfg: Config = cfg):
        self.n = n_samples
        self.cfg = cfg

        # Create a handful of "concepts" that loosely tie image intensity to certain tokens
        self.concepts = [
            {"img_mean": 0.2, "token_ids": list(range(10))},
            {"img_mean": 0.5, "token_ids": list(range(200, 220))},
            {"img_mean": 0.8, "token_ids": list(range(400, 420))},
        ]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        concept = random.choice(self.concepts)
        # Image with a certain mean intensity
        img = (
            torch.randn(3, self.cfg.img_size, self.cfg.img_size) * 0.2
            + concept["img_mean"]
        )
        # Text with tokens from the concept range + some noise tokens
        length = random.randint(self.cfg.max_len // 2, self.cfg.max_len)
        tokens = torch.randint(0, self.cfg.vocab_size, (length,))
        mask = torch.ones(length, dtype=torch.bool)
        # Overwrite a few positions with concept tokens
        for _ in range(length // 4):
            tokens[random.randrange(length)] = random.choice(concept["token_ids"])
        return img.clamp(0, 1), tokens, mask


def collate(batch):
    """Pad variable-length token sequences to the same length within a batch."""
    imgs, seqs, masks = zip(*batch)
    imgs = torch.stack(imgs, dim=0)  # (B, 3, H, W)

    max_len = max(s.size(0) for s in seqs)
    padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
    attn_mask = torch.zeros(len(seqs), max_len, dtype=torch.bool)
    for i, (s, m) in enumerate(zip(seqs, masks)):
        padded[i, : s.size(0)] = s
        attn_mask[i, : s.size(0)] = True
    return imgs, padded, attn_mask


class TinyImageEncoder(nn.Module):
    """A very small CNN + global average pooling -> d_model."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.proj = nn.Linear(256, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)  # (B, 256, H/8, W/8)
        h = h.mean(dim=[2, 3])  # GAP -> (B, 256)
        return self.proj(h)  # (B, d_model)


class TinyTextEncoder(nn.Module):
    """Embedding + mean pooling -> d_model."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # tokens: (B, L), mask: (B, L) with True for valid tokens
        x = self.embed(tokens)  # (B, L, d_model)
        mask = mask.unsqueeze(-1).float()
        x = (x * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1.0))
        return x  # (B, d_model)


# MoE
class Expert(nn.Module):
    """Simple 2-layer MLP expert."""

    def __init__(self, d_model: int, hidden: int = 512):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class TopKGate(nn.Module):
    """Top-k gating network returning weighted sum across selected experts.
    Also returns auxiliary stats for load balancing loss.
    """

    def __init__(self, d_model: int, num_experts: int, k: int):
        super().__init__()
        assert k <= num_experts
        self.w_gating = nn.Linear(d_model, num_experts)
        self.num_experts = num_experts
        self.k = k

    def forward(self, x: torch.Tensor):
        # x: (B, d_model)
        logits = self.w_gating(x)  # (B, E)
        gate_probs = F.softmax(logits, dim=-1)  # (B, E)

        # Top-k selection per sample
        topk_vals, topk_idx = torch.topk(gate_probs, k=self.k, dim=-1)  # (B, k)
        # Normalize selected weights to sum to 1 (optional but common)
        norm_weights = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)

        # Build a (B, E) sparse-like mask for stats (not used for compute)
        hard_mask = torch.zeros_like(gate_probs)
        hard_mask.scatter_(1, topk_idx, 1.0)

        return topk_idx, norm_weights, gate_probs, hard_mask


class MoELayer(nn.Module):
    """Shared experts + top-k gating. Computes weighted sum of expert outputs."""

    def __init__(self, d_model: int, num_experts: int, top_k: int, hidden: int = 512):
        super().__init__()
        self.experts = nn.ModuleList(
            [Expert(d_model, hidden) for _ in range(num_experts)]
        )
        self.gate = TopKGate(d_model, num_experts, top_k)

    def forward(self, x: torch.Tensor):
        # x: (B, d_model)
        B, D = x.shape
        topk_idx, norm_weights, gate_probs, hard_mask = self.gate(
            x
        )  # idx: (B, k), weights: (B, k)

        # Collect expert outputs for selected experts only
        expert_outputs = []
        for rank in range(topk_idx.size(1)):  # over k
            idx = topk_idx[:, rank]  # (B,)
            # Process each sample by its selected expert
            # Efficient batched routing is more complex; here we use a simple loop for clarity.
            chunk_out = torch.empty(B, D, device=x.device, dtype=x.dtype)
            for e in range(len(self.experts)):
                # mask for samples choosing expert e at this rank
                sel = idx == e
                if sel.any():
                    chunk_out[sel] = self.experts[e](x[sel])
            expert_outputs.append(chunk_out)  # list of (B, D)

        # Weighted sum across k experts
        stacked = torch.stack(expert_outputs, dim=1)  # (B, k, D)
        weights = norm_weights.unsqueeze(-1)  # (B, k, 1)
        y = (stacked * weights).sum(dim=1)  # (B, D)

        # Auxiliary load-balancing loss (KL to uniform over experts)
        mean_probs = gate_probs.mean(dim=0)  # (E,)
        uniform = torch.full_like(mean_probs, 1.0 / mean_probs.numel())
        aux_loss = F.kl_div(mean_probs.log(), uniform, reduction="batchmean")

        return y, aux_loss, gate_probs, hard_mask


class MultiModalMoE(nn.Module):
    """Encodes image and text, applies MoE to each, projects to shared space, computes contrastive logits."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.img_enc = TinyImageEncoder(cfg.d_model)
        self.txt_enc = TinyTextEncoder(cfg.vocab_size, cfg.d_model)

        # Share one MoE across modalities (optional). You could also create two separate MoEs.
        self.moe = MoELayer(cfg.d_model, cfg.num_experts, cfg.top_k, hidden=512)

        # Projection heads to embedding space used for contrastive learning
        self.img_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.txt_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # Temperature parameter (log for stability)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / cfg.temperature)))

    def forward(self, img: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor):
        # Encode modalities
        h_img = self.img_enc(img)  # (B, d)
        h_txt = self.txt_enc(tokens, mask)  # (B, d)

        # Apply MoE to each modality
        z_img, aux_img, gp_img, _ = self.moe(h_img)  # (B, d)
        z_txt, aux_txt, gp_txt, _ = self.moe(h_txt)  # (B, d)

        # Projections
        e_img = F.normalize(self.img_proj(z_img), dim=-1)  # (B, d)
        e_txt = F.normalize(self.txt_proj(z_txt), dim=-1)  # (B, d)

        # Contrastive logits
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits_per_img = logit_scale * e_img @ e_txt.t()  # (B, B)
        logits_per_txt = logits_per_img.t()

        aux_loss = (aux_img + aux_txt) * 0.5
        return logits_per_img, logits_per_txt, aux_loss


def clip_style_loss(
    logits_per_img: torch.Tensor, logits_per_txt: torch.Tensor
) -> torch.Tensor:
    """Symmetric cross-entropy over image->text and text->image alignments."""
    B = logits_per_img.size(0)
    target = torch.arange(B, device=logits_per_img.device)
    loss_i = F.cross_entropy(logits_per_img, target)
    loss_t = F.cross_entropy(logits_per_txt, target)
    return (loss_i + loss_t) * 0.5


def train(cfg: Config):
    device = torch.device(cfg.device)
    ds = ToyImageTextDataset(n_samples=4096, cfg=cfg)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        drop_last=True,
    )

    model = MultiModalMoE(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    print(f"Training on {device} for {cfg.epochs} epochs...")

    for epoch in range(cfg.epochs):
        model.train()
        total, total_aux = 0.0, 0.0
        for step, (imgs, tokens, mask) in enumerate(dl, 1):
            imgs, tokens, mask = imgs.to(device), tokens.to(device), mask.to(device)

            logits_i, logits_t, aux = model(imgs, tokens, mask)
            loss_main = clip_style_loss(logits_i, logits_t)
            loss = loss_main + 0.01 * aux  # small weight for load-balancing

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss_main.item()
            total_aux += aux.item()

            if step % 50 == 0:
                with torch.no_grad():
                    # Report sparsity/utilization: fraction of max gate chosen per batch (rough signal)
                    # Note: We don't expose gate stats here to keep code compact.
                    print(
                        f"Epoch {epoch+1} Step {step}: loss={total/step:.4f} aux={total_aux/step:.4f}"
                    )

        print(
            f"Epoch {epoch+1} done. Avg loss={total/len(dl):.4f}, aux={total_aux/len(dl):.4f}"
        )

    # Quick sanity check: similarity of matched pairs vs mismatched
    model.eval()
    imgs, tokens, mask = next(iter(dl))
    imgs, tokens, mask = imgs.to(device), tokens.to(device), mask.to(device)
    with torch.no_grad():
        logits_i, _, _ = model(imgs, tokens, mask)
        pos = logits_i.diag().mean().item()
        neg = (logits_i.sum() - logits_i.diag().sum()) / (
            logits_i.numel() - imgs.size(0)
        )
    print(f"Avg positive logit: {pos:.3f}, avg negative logit: {neg:.3f}")


if __name__ == "__main__":
    train(cfg)
