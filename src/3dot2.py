"""
3.2  v-Prediction Flow Matching at D = 2
=========================================
Schedule  : cosine  –  α_t = cos(πt/2),  σ_t = sin(πt/2),  t ∈ [0,1]
            t=1 → pure noise,  t=0 → data
Interpolant: x_t = α_t · x_data + σ_t · ε
v-target   : v = α_t · ε − σ_t · x_data
v-loss     : MSE(v_θ(x_t, t), v)
ODE        : dx_t/dt = (π/2) · v_θ(x_t, t)   [integrate t: 1 → 0]

Hyperparameters (D = 2)
-----------------------
  Model     : MLP, sinusoidal time-emb (dim 128), 3 × hidden-256, SiLU
  Optimizer : Adam, lr = 1e-3
  Batch size: 1024
  Training  : 25 000 steps
  Sampling  : Euler ODE, 50 steps
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataloader import ToyDiffusionDataset, get_dataloader

# ── Sinusoidal time embedding (DiT-style) ─────────────────────────────────────

def sinusoidal_embedding(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    """Map scalar time t ∈ [0,1] to a (B, dim) sinusoidal embedding."""
    assert dim % 2 == 0
    half = dim // 2
    # log-spaced frequencies
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
    )
    args = t.float()[:, None] * freqs[None]          # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


# ── Model ─────────────────────────────────────────────────────────────────────

class VPredModel(nn.Module):
    """Simple MLP that predicts v given (x_t, t)."""

    def __init__(self, data_dim: int = 2, time_emb_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: (B, D),  t: (B,)  →  v_pred: (B, D)"""
        t_emb = sinusoidal_embedding(t, self.time_emb_dim)
        return self.net(torch.cat([x, t_emb], dim=-1))


# ── v-prediction loss ─────────────────────────────────────────────────────────

def v_loss(model: VPredModel, x1: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Compute the v-prediction MSE loss for a batch of data samples x1."""
    B, D = x1.shape
    t = torch.rand(B, device=device)                          # t ~ U[0,1]
    eps = torch.randn(B, D, device=device)                    # ε ~ N(0,I)

    alpha_t = torch.cos(0.5 * math.pi * t).unsqueeze(1)      # (B,1)
    sigma_t = torch.sin(0.5 * math.pi * t).unsqueeze(1)      # (B,1)

    x_t = alpha_t * x1 + sigma_t * eps                       # interpolant
    v_target = alpha_t * eps - sigma_t * x1                  # v-prediction target

    v_pred = model(x_t, t)
    return ((v_pred - v_target) ** 2).mean()


# ── Euler ODE sampler ─────────────────────────────────────────────────────────

@torch.no_grad()
def euler_sample(
    model: VPredModel,
    n_samples: int,
    dim: int,
    device: torch.device,
    n_steps: int = 50,
) -> torch.Tensor:
    """
    Generate samples by integrating the learned ODE from t=1 → t=0.
    ODE: dx_t/dt = (π/2) · v_θ(x_t, t)
    Euler step: x_{t+Δt} = x_t + Δt · (π/2) · v_θ(x_t, t),  Δt < 0
    """
    model.eval()
    x = torch.randn(n_samples, dim, device=device)            # x at t=1 is pure noise
    ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device)  # t: 1→0

    for i in range(n_steps):
        t_curr = ts[i]
        dt = ts[i + 1] - t_curr                               # negative
        t_batch = t_curr.expand(n_samples)
        v = model(x, t_batch)
        x = x + dt * (math.pi / 2.0) * v

    model.train()
    return x


# ── Training loop ─────────────────────────────────────────────────────────────

def train_model(
    dataset_name: str,
    dim: int = 2,
    n_steps: int = 25_000,
    batch_size: int = 1024,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> VPredModel:
    loader = get_dataloader(dataset_name, dim=dim, batch_size=batch_size, shuffle=True)
    data_iter = iter(loader)

    model = VPredModel(data_dim=dim).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    print(f"\n=== Training on '{dataset_name}' (D={dim}) ===")
    for step in range(1, n_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        batch = batch.float().to(device)
        loss = v_loss(model, batch, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 5000 == 0:
            print(f"  step {step:6d}/{n_steps}  loss={loss.item():.5f}")

    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    DIM = 2
    DATASETS = ["swiss_roll", "gaussians", "circles"]
    N_SAMPLE_STEPS = 50

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("v-Prediction Flow Matching  (D=2, cosine schedule)", fontsize=14)

    for col, name in enumerate(DATASETS):
        # Ground truth
        ds = ToyDiffusionDataset(name=name, dim=DIM)
        gt = ds.data.numpy()

        # Train
        model = train_model(name, dim=DIM, device=device)

        # Sample
        samples = euler_sample(model, n_samples=len(gt), dim=DIM,
                               device=device, n_steps=N_SAMPLE_STEPS)
        samples = samples.cpu().numpy()

        # Plot ground truth (top row)
        axes[0, col].scatter(gt[:, 0], gt[:, 1], s=2, alpha=0.4, c="steelblue")
        axes[0, col].set_title(f"{name} — ground truth")
        axes[0, col].set_aspect("equal")

        # Plot generated (bottom row)
        axes[1, col].scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.4, c="coral")
        axes[1, col].set_title(f"{name} — generated (v-pred, {N_SAMPLE_STEPS} steps)")
        axes[1, col].set_aspect("equal")

    plt.tight_layout()
    out_path = Path(__file__).resolve().parent.parent / "part2_v_prediction_D2.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
