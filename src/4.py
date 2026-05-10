"""
4.py
Part 2 / Part 4 – Flow Matching Parameterization
================================================

This script runs all 36 experiments required by the assignment:

    3 datasets × 3 dimensions × 4 prediction/loss combinations

Datasets:
    swiss_roll, gaussians, circles

Dimensions:
    D = 2, 8, 32

Prediction/loss combinations:
    x-pred + x-loss
    x-pred + v-loss
    v-pred + x-loss
    v-pred + v-loss

Forward process:
    z_t = (1 - t) * x + t * eps,    t ∈ [0, 1]

where:
    t = 0 -> clean data x
    t = 1 -> Gaussian noise eps

True velocity:
    v = eps - x

Since:
    z_t = (1 - t)x + t eps
        = x + t(eps - x)
        = x + t v

Conversion formulas:
    x -> v:
        v = (z_t - x) / t

    v -> x:
        x = z_t - t * v

Sampling:
    Start from z_1 ~ N(0, I), integrate Euler ODE from t = 1 to t = 0.

    dz_t / dt = v_theta(z_t, t)

    If model predicts v:
        v_theta = model(z_t, t)

    If model predicts x:
        x_hat = model(z_t, t)
        v_theta = (z_t - x_hat) / t

Outputs:
    part4_results/
        figures/
            swiss_roll_D2.png
            swiss_roll_D8.png
            ...
        raw/
            swiss_roll_D2_xpred_xloss.npz
            ...
        losses/
            swiss_roll_D2_xpred_xloss_loss.csv
            ...
        summary.csv

Hyperparameters:
    Model      : 5 hidden layers, hidden width 256, ReLU
    Time emb   : 128-d sinusoidal embedding
    Optimizer  : Adam, lr = 1e-3
    Batch size : 1024
    Training   : 25,000 steps
    Sampling   : Euler ODE, 50 steps
    t clipping : [1e-3, 1 - 1e-3]
"""

import csv
import math
import sys
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


# =============================================================================
# Import dataloader
# =============================================================================

# Assumption:
#   4.py is in the same directory as dataloader.py.
# If your file structure is different, adjust this path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataloader import ToyDiffusionDataset, get_dataloader


# =============================================================================
# Hyperparameters
# =============================================================================

T_EPS = 1e-3

N_TRAIN = 25_000
BATCH_SIZE = 1024
LR = 1e-3

N_SAMPLE_STEPS = 50

TIME_EMB_DIM = 128
HIDDEN_DIM = 256

PRED_TYPES = ["x", "v"]
LOSS_TYPES = ["x", "v"]

DATASETS = ["swiss_roll", "gaussians", "circles"]
DIMS = [2, 8, 32]

LOSS_LOG_INTERVAL = 500


# =============================================================================
# Sinusoidal time embedding
# =============================================================================

def sinusoidal_embedding(t: torch.Tensor, dim: int = TIME_EMB_DIM) -> torch.Tensor:
    """
    Map scalar time t ∈ [0, 1] to a sinusoidal embedding of shape (B, dim).

    For dim = 128:
        64 sine features + 64 cosine features.

    Frequencies:
        omega_i = exp(- i * ln(10000) / (k - 1)), i = 0, ..., k-1
        where k = dim / 2.
    """
    assert dim % 2 == 0

    half = dim // 2

    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / (half - 1)
    )

    args = t.float()[:, None] * freqs[None, :]

    emb = torch.cat(
        [
            torch.sin(args),
            torch.cos(args),
        ],
        dim=-1,
    )

    return emb


# =============================================================================
# Model
# =============================================================================

class FlowModel(nn.Module):
    """
    Assignment-specified MLP.

    Input:
        [z_t ; e_t] ∈ R^{D + 128}

    Architecture:
        Linear(D + 128 -> 256) + ReLU
        Linear(256 -> 256) + ReLU
        Linear(256 -> 256) + ReLU
        Linear(256 -> 256) + ReLU
        Linear(256 -> 256) + ReLU
        Linear(256 -> D)

    This is 5 hidden layers and 6 Linear layers in total.

    The architecture is identical for all four configurations.

    pred_type determines whether the raw model output is interpreted as:
        x_hat or v_hat

    loss_type determines whether MSE is computed in:
        x-space or v-space
    """

    def __init__(
        self,
        data_dim: int,
        time_emb_dim: int = TIME_EMB_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ):
        super().__init__()

        self.time_emb_dim = time_emb_dim

        self.net = nn.Sequential(
            nn.Linear(data_dim + time_emb_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        z: shape (B, D)
        t: shape (B,)

        returns:
            raw prediction of shape (B, D)
        """
        t_emb = sinusoidal_embedding(t, self.time_emb_dim)
        inp = torch.cat([z, t_emb], dim=-1)
        return self.net(inp)


# =============================================================================
# Loss
# =============================================================================

def compute_loss(
    model: FlowModel,
    x: torch.Tensor,
    pred_type: str,
    loss_type: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute the training loss for one of the four combinations:

        x-pred + x-loss
        x-pred + v-loss
        v-pred + x-loss
        v-pred + v-loss
    """

    assert pred_type in ["x", "v"]
    assert loss_type in ["x", "v"]

    B, D = x.shape

    # Uniform time sampling.
    # We clip globally for numerical stability.
    # This is especially important for x-pred -> v conversion,
    # where v_hat = (z_t - x_hat) / t.
    t = torch.rand(B, device=device)
    t = t.clamp(T_EPS, 1.0 - T_EPS)

    t_col = t.unsqueeze(1)

    eps = torch.randn(B, D, device=device)

    # Linear interpolant:
    # z_t = (1 - t) x + t eps
    z_t = (1.0 - t_col) * x + t_col * eps

    # True velocity:
    # v = eps - x
    v = eps - x

    raw = model(z_t, t)

    # Interpret raw model output.
    if pred_type == "x":
        x_hat = raw
        v_hat = (z_t - x_hat) / t_col

    else:
        v_hat = raw
        x_hat = z_t - t_col * v_hat

    # Compute loss in the requested space.
    if loss_type == "x":
        loss = ((x_hat - x) ** 2).mean()
    else:
        loss = ((v_hat - v) ** 2).mean()

    return loss


# =============================================================================
# Euler ODE sampler
# =============================================================================

@torch.no_grad()
def euler_sample(
    model: FlowModel,
    n_samples: int,
    dim: int,
    pred_type: str,
    device: torch.device,
    n_steps: int = N_SAMPLE_STEPS,
) -> torch.Tensor:
    """
    Generate samples by integrating from t = 1 to t = 0.

    Initial condition:
        z_1 ~ N(0, I)

    ODE:
        dz_t / dt = v_theta(z_t, t)

    For v-prediction:
        v_theta = model(z_t, t)

    For x-prediction:
        x_hat = model(z_t, t)
        v_theta = (z_t - x_hat) / t
    """

    assert pred_type in ["x", "v"]

    model.eval()

    z = torch.randn(n_samples, dim, device=device)

    ts = torch.linspace(
        1.0,
        0.0,
        n_steps + 1,
        device=device,
    )

    for i in range(n_steps):
        t_curr = ts[i]
        t_next = ts[i + 1]

        dt = t_next - t_curr  # negative, because t goes from 1 to 0

        t_batch = t_curr.expand(n_samples)

        raw = model(z, t_batch)

        if pred_type == "x":
            t_safe = t_curr.clamp(min=T_EPS)
            v_hat = (z - raw) / t_safe
        else:
            v_hat = raw

        z = z + dt * v_hat

    model.train()

    return z


# =============================================================================
# Training
# =============================================================================

def train_model(
    dataset_name: str,
    dim: int,
    pred_type: str,
    loss_type: str,
    device: torch.device,
    n_steps: int = N_TRAIN,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
) -> tuple[FlowModel, list[tuple[int, float]]]:
    """
    Train one model for one configuration:

        dataset_name ∈ {swiss_roll, gaussians, circles}
        dim          ∈ {2, 8, 32}
        pred_type    ∈ {x, v}
        loss_type    ∈ {x, v}

    Returns:
        model, loss_log

    loss_log:
        list of (step, loss_value)
    """

    loader = get_dataloader(
        dataset_name,
        dim=dim,
        batch_size=batch_size,
        shuffle=True,
    )

    data_iter = iter(loader)

    model = FlowModel(data_dim=dim).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    loss_log: list[tuple[int, float]] = []

    print(
        f"\n=== Training: dataset={dataset_name}, D={dim}, "
        f"{pred_type}-pred + {loss_type}-loss ==="
    )

    for step in range(1, n_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        batch = batch.float().to(device)

        loss = compute_loss(
            model=model,
            x=batch,
            pred_type=pred_type,
            loss_type=loss_type,
            device=device,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % LOSS_LOG_INTERVAL == 0 or step == 1 or step == n_steps:
            loss_value = float(loss.item())
            loss_log.append((step, loss_value))

        if step % 5000 == 0:
            print(
                f"    step {step:6d}/{n_steps} "
                f"loss={loss.item():.6f}"
            )

    return model, loss_log


# =============================================================================
# Save helpers
# =============================================================================

def save_loss_log(
    loss_log: list[tuple[int, float]],
    loss_path: Path,
) -> None:
    """
    Save loss log as CSV.
    """
    with loss_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])
        for step, loss_value in loss_log:
            writer.writerow([step, loss_value])


def save_raw_result(
    raw_path: Path,
    gt_raw: np.ndarray,
    gt_2d: np.ndarray,
    samples_raw: np.ndarray,
    samples_2d: np.ndarray,
    dataset_name: str,
    dim: int,
    pred_type: str,
    loss_type: str,
    final_loss: float,
) -> None:
    """
    Save raw arrays and minimal metadata as compressed npz.
    """
    np.savez_compressed(
        raw_path,
        gt_raw=gt_raw,
        gt_2d=gt_2d,
        samples_raw=samples_raw,
        samples_2d=samples_2d,
        dataset_name=np.array(dataset_name),
        dim=np.array(dim),
        pred_type=np.array(pred_type),
        loss_type=np.array(loss_type),
        final_loss=np.array(final_loss),
    )


# =============================================================================
# Visualization
# =============================================================================

def scatter2d(
    ax,
    data: np.ndarray,
    color: str,
    title: str,
) -> None:
    """
    Plot 2D scatter.

    For D = 8 or D = 32, data should already have been projected to 2D
    using the dataset-provided to_2d function.
    """
    ax.scatter(
        data[:, 0],
        data[:, 1],
        s=2,
        alpha=0.4,
        c=color,
        rasterized=True,
    )

    ax.set_title(title, fontsize=8)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)


def plot_dataset_dim_figure(
    dataset_name: str,
    dim: int,
    gt_2d: np.ndarray,
    generated_by_combo: dict[tuple[str, str], np.ndarray],
    fig_path: Path,
) -> None:
    """
    Save one figure for a fixed dataset and dimension.

    Figure layout:
        2 rows × 4 columns

        top row:
            ground truth repeated for each combo

        bottom row:
            generated samples for each combo
    """
    combos = list(product(PRED_TYPES, LOSS_TYPES))

    combo_labels = [
        f"{pred_type}-pred\n{loss_type}-loss"
        for pred_type, loss_type in combos
    ]

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(18, 9),
        constrained_layout=True,
    )

    fig.suptitle(
        f"Flow Matching Parameterization – {dataset_name}, D={dim}\n"
        f"Top: ground truth | Bottom: generated | "
        f"Euler ODE, {N_SAMPLE_STEPS} steps",
        fontsize=11,
    )

    for col, ((pred_type, loss_type), clabel) in enumerate(
        zip(combos, combo_labels)
    ):
        samples_2d = generated_by_combo[(pred_type, loss_type)]

        scatter2d(
            axes[0, col],
            gt_2d,
            color="steelblue",
            title=f"{clabel}\nGround truth",
        )

        scatter2d(
            axes[1, col],
            samples_2d,
            color="coral",
            title=f"{clabel}\nGenerated",
        )

    fig.savefig(fig_path, dpi=130)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    root_dir = Path(__file__).resolve().parent
    out_dir = root_dir / "part4_results"

    fig_dir = out_dir / "figures"
    raw_dir = out_dir / "raw"
    loss_dir = out_dir / "losses"

    fig_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    loss_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.csv"

    combos = list(product(PRED_TYPES, LOSS_TYPES))

    summary_rows: list[dict[str, str | int | float]] = []

    # Total:
    #   3 dimensions × 3 datasets × 4 combos = 36 experiments.
    #
    # We save:
    #   - one raw .npz per experiment
    #   - one loss .csv per experiment
    #   - one figure per dataset/dim, containing 4 generated panels
    #
    # This gives:
    #   - 36 raw files
    #   - 36 loss files
    #   - 9 report-ready figures
    #   - 36 generated visualizations in total

    for dim in DIMS:
        for dataset_name in DATASETS:
            print("\n" + "=" * 90)
            print(f"Dataset: {dataset_name}, D={dim}")
            print("=" * 90)

            ds = ToyDiffusionDataset(
                name=dataset_name,
                dim=dim,
            )

            gt_raw = ds.data.numpy()
            gt_2d = ds.to_2d(gt_raw)

            generated_by_combo: dict[tuple[str, str], np.ndarray] = {}

            for pred_type, loss_type in combos:
                tag = f"{dataset_name}_D{dim}_{pred_type}pred_{loss_type}loss"

                raw_path = raw_dir / f"{tag}.npz"
                loss_path = loss_dir / f"{tag}_loss.csv"

                model, loss_log = train_model(
                    dataset_name=dataset_name,
                    dim=dim,
                    pred_type=pred_type,
                    loss_type=loss_type,
                    device=device,
                    n_steps=N_TRAIN,
                    batch_size=BATCH_SIZE,
                    lr=LR,
                )

                final_loss = float(loss_log[-1][1])

                samples_raw_tensor = euler_sample(
                    model=model,
                    n_samples=len(gt_raw),
                    dim=dim,
                    pred_type=pred_type,
                    device=device,
                    n_steps=N_SAMPLE_STEPS,
                )

                samples_raw = samples_raw_tensor.cpu().numpy()
                samples_2d = ds.to_2d(samples_raw)

                generated_by_combo[(pred_type, loss_type)] = samples_2d

                save_raw_result(
                    raw_path=raw_path,
                    gt_raw=gt_raw,
                    gt_2d=gt_2d,
                    samples_raw=samples_raw,
                    samples_2d=samples_2d,
                    dataset_name=dataset_name,
                    dim=dim,
                    pred_type=pred_type,
                    loss_type=loss_type,
                    final_loss=final_loss,
                )

                save_loss_log(
                    loss_log=loss_log,
                    loss_path=loss_path,
                )

                summary_rows.append(
                    {
                        "dataset": dataset_name,
                        "dim": dim,
                        "pred_type": pred_type,
                        "loss_type": loss_type,
                        "final_loss": final_loss,
                        "raw_path": str(raw_path),
                        "loss_path": str(loss_path),
                    }
                )

                print(f"    Saved raw  -> {raw_path}")
                print(f"    Saved loss -> {loss_path}")

            fig_path = fig_dir / f"{dataset_name}_D{dim}.png"

            plot_dataset_dim_figure(
                dataset_name=dataset_name,
                dim=dim,
                gt_2d=gt_2d,
                generated_by_combo=generated_by_combo,
                fig_path=fig_path,
            )

            print(f"\nSaved figure -> {fig_path}")

    # Save summary.csv
    with summary_path.open("w", newline="") as f:
        fieldnames = [
            "dataset",
            "dim",
            "pred_type",
            "loss_type",
            "final_loss",
            "raw_path",
            "loss_path",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in summary_rows:
            writer.writerow(row)

    print("\n" + "=" * 90)
    print("All 36 experiments completed.")
    print(f"Results saved in: {out_dir}")
    print(f"Figures: {fig_dir}")
    print(f"Raw samples: {raw_dir}")
    print(f"Loss logs: {loss_dir}")
    print(f"Summary: {summary_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()