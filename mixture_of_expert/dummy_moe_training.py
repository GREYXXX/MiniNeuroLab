import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import math
from typing import Tuple, Optional


class Expert(nn.Module):
    """Individual expert network - a simple MLP"""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SparseGatingNetwork(nn.Module):
    """
    Gating network that routes inputs to top-k experts
    Implements noisy top-k gating from Switch Transformer paper
    """

    def __init__(
        self, input_dim: int, num_experts: int, top_k: int = 2, noise_std: float = 0.1
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std

        # Gate network - outputs logits for each expert
        self.gate = nn.Linear(input_dim, num_experts)

        # Noise network for exploration (optional)
        self.noise_gate = nn.Linear(input_dim, num_experts) if noise_std > 0 else None

    def forward(
        self, x: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]

        Returns:
            gates: Softmax probabilities for selected experts [batch_size, seq_len, top_k]
            indices: Indices of selected experts [batch_size, seq_len, top_k]
            raw_gates: Raw logits before selection [batch_size, seq_len, num_experts]
        """
        batch_size, seq_len = x.shape[:2]

        # Compute gate logits
        raw_gates = self.gate(x)  # [batch_size, seq_len, num_experts]

        # Add noise during training for exploration
        if training and self.noise_gate is not None:
            noise = torch.randn_like(raw_gates) * self.noise_std
            noise_logits = self.noise_gate(x)
            raw_gates += noise * F.softplus(noise_logits)

        # Select top-k experts
        topk_logits, topk_indices = torch.topk(raw_gates, self.top_k, dim=-1)

        # Compute gates (softmax over top-k)
        gates = F.softmax(topk_logits, dim=-1)

        return gates, topk_indices, raw_gates


class MixtureOfExperts(nn.Module):
    """
    Complete Mixture of Experts module with load balancing
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        top_k: int = 2,
        dropout: float = 0.1,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight

        # Create experts
        self.experts = nn.ModuleList(
            [
                Expert(input_dim, hidden_dim, output_dim, dropout)
                for _ in range(num_experts)
            ]
        )

        # Gating network
        self.gate = SparseGatingNetwork(input_dim, num_experts, top_k)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through MoE

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]

        Returns:
            output: Mixed expert outputs [batch_size, seq_len, output_dim]
            aux_loss_dict: Dictionary containing auxiliary losses
        """
        batch_size, seq_len, input_dim = x.shape

        # Get gating decisions
        gates, expert_indices, raw_gates = self.gate(x, self.training)
        # gates: [batch_size, seq_len, top_k]
        # expert_indices: [batch_size, seq_len, top_k]
        # raw_gates: [batch_size, seq_len, num_experts]

        # Initialize output
        output = torch.zeros(
            batch_size,
            seq_len,
            self.experts[0].net[-1].out_features,
            device=x.device,
            dtype=x.dtype,
        )

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find positions where this expert is selected
            expert_mask = expert_indices == expert_idx  # [batch_size, seq_len, top_k]

            if not expert_mask.any():
                continue

            # Get positions and corresponding gate weights
            positions = torch.where(expert_mask)
            batch_indices, seq_indices, topk_indices = positions

            # Extract inputs for this expert
            expert_inputs = x[batch_indices, seq_indices]  # [num_tokens, input_dim]

            # Get corresponding gate weights
            expert_gates = gates[
                batch_indices, seq_indices, topk_indices
            ]  # [num_tokens]

            # Forward through expert
            if expert_inputs.size(0) > 0:
                expert_output = self.experts[expert_idx](expert_inputs)

                # Weight by gates and accumulate
                weighted_output = expert_output * expert_gates.unsqueeze(-1)
                output[batch_indices, seq_indices] += weighted_output

        # Compute auxiliary losses
        aux_losses = self._compute_aux_losses(gates, expert_indices, raw_gates)

        return output, aux_losses

    def _compute_aux_losses(
        self, gates: torch.Tensor, expert_indices: torch.Tensor, raw_gates: torch.Tensor
    ) -> dict:
        """Compute load balancing and other auxiliary losses"""

        # Load balancing loss - encourages uniform expert usage
        # Based on Switch Transformer paper
        gate_probs = F.softmax(raw_gates, dim=-1)  # [batch_size, seq_len, num_experts]

        # Compute fraction of tokens assigned to each expert
        expert_counts = torch.zeros_like(gate_probs)
        for k in range(self.top_k):
            expert_counts.scatter_add_(
                -1, expert_indices[:, :, k : k + 1], gates[:, :, k : k + 1]
            )

        # Average over batch and sequence dimensions
        avg_gate_probs = gate_probs.mean(dim=[0, 1])  # [num_experts]
        avg_expert_counts = expert_counts.mean(dim=[0, 1])  # [num_experts]

        # Load balance loss: minimize variance in expert usage
        load_balance_loss = self.num_experts * torch.sum(
            avg_gate_probs * avg_expert_counts
        )

        return {
            "load_balance_loss": self.load_balance_weight * load_balance_loss,
            "expert_usage": avg_expert_counts,
            "gate_entropy": -torch.sum(
                avg_gate_probs * torch.log(avg_gate_probs + 1e-8)
            ),
        }


class MoETransformerLayer(nn.Module):
    """
    Transformer layer with MoE instead of traditional FFN
    Demonstrates how MoE integrates into existing architectures
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_experts: int,
        expert_hidden_dim: int,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # MoE instead of traditional FFN
        self.moe = MixtureOfExperts(
            input_dim=d_model,
            hidden_dim=expert_hidden_dim,
            output_dim=d_model,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            src_mask: Attention mask

        Returns:
            output: Transformed tensor [batch_size, seq_len, d_model]
            aux_losses: Auxiliary losses from MoE
        """
        # Self attention with residual connection
        attn_output, _ = self.self_attn(x, x, x, attn_mask=src_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # MoE with residual connection
        moe_output, aux_losses = self.moe(x)
        output = self.norm2(x + self.dropout(moe_output))

        return output, aux_losses


# Example usage and training loop
def demonstrate_moe():
    """Demonstration of MoE training and inference"""

    # Model parameters
    batch_size = 4
    seq_len = 32
    d_model = 256
    num_experts = 8
    top_k = 2
    num_heads = 8
    expert_hidden_dim = 512

    # Create model
    model = MoETransformerLayer(
        d_model=d_model,
        num_heads=num_heads,
        num_experts=num_experts,
        expert_hidden_dim=expert_hidden_dim,
        top_k=top_k,
    )

    # Generate dummy data
    x = torch.randn(batch_size, seq_len, d_model)
    target = torch.randn(batch_size, seq_len, d_model)

    # Forward pass
    output, aux_losses = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Load balance loss: {aux_losses['load_balance_loss']:.4f}")
    print(f"Gate entropy: {aux_losses['gate_entropy']:.4f}")
    print(f"Expert usage: {aux_losses['expert_usage'].tolist()}")

    # Training step
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Main task loss (e.g., reconstruction)
    main_loss = F.mse_loss(output, target)

    # Total loss includes auxiliary losses
    total_loss = main_loss + aux_losses["load_balance_loss"]

    print(f"\nMain loss: {main_loss:.4f}")
    print(f"Total loss: {total_loss:.4f}")

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()

    # Gradient clipping (important for MoE training)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    print("\nTraining step completed!")

    # Inference mode
    model.eval()
    with torch.no_grad():
        eval_output, eval_aux = model(x)
        print(f"\nEval expert usage: {eval_aux['expert_usage'].tolist()}")


if __name__ == "__main__":
    demonstrate_moe()
