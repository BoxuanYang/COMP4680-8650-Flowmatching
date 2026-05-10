"""
6.py
Part 4 – One-Step Generation
============================

6.1 Sampling Efficiency
-----------------------
Using the best Part-2 model, i.e. x-prediction with x-loss at D=32,
evaluate Euler ODE sampling quality across step counts:

    1, 2, 5, 10, 20, 50, 100, 200

One figure is generated per dataset.

6.2 MeanFlow
------------
Implement MeanFlow using an additional horizon input h = t - r.
The MeanFlow model predicts the interval-average velocity:

    u_theta(z_t, r, t)

The training loss uses torch.func.jvp to compute the total derivative:

    d/dt u_theta(z_t, r, t)

with tangent vector:

    (dz_t/dt, dr/dt, dt/dt) = (v_t, 0, 1)

where:

    v_t = eps - x

The MeanFlow target is:

    u_tgt = v_t - (t-r) * d/dt u_theta(z_t, r, t)

and the loss is:

    || u_theta(z_t, r, t) - stopgrad(u_tgt) ||^2

Training uses a flow matching ratio of 0.5:
    - 50% h = 0, equivalent to standard flow matching
    - 50% h > 0, mean-velocity training

Sampling uses:

    z_r = z_t - (t-r) * u_theta(z_t, r, t)

Outputs
-------
part6_results/
    figures/
        6.1_<dataset>_D32_steps.png
        6.2_<dataset>_<k>step.png
        6.compare_<dataset>_D32.png

    raw/
        6.1_<dataset>_D32_steps.npz
        6.2_<dataset>_D32_mf.npz

    losses/
        6.1_<dataset>_D32_loss.csv
        6.2_<dataset>_D32_mf_loss.csv

    summary.csv

Datasets
--------
    swiss_roll, gaussians, circles

Dimension
---------
    D = 32
"""

import csv
import math
import sys
import time
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataloader import ToyDiffusionDataset, get_dataloader


# =============================================================================
# Hyperparameters
# =============================================================================

T_EPS = 1e-3

N_TRAIN_FM = 25_000
N_TRAIN_MF = 25_000

BATCH_SIZE = 1024
LR = 1e-3

TIME_EMB = 128
HIDDEN = 256
DIM = 32

FM_RATIO = 0.5

DATASETS = ["swiss_roll", "gaussians", "circles"]

STEP_COUNTS_61 = [1, 2, 5, 10, 20, 50, 100, 200]
STEP_COUNTS_62 = [1, 2, 5]

LOG_EVERY = 500

SEED = 42


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# Utilities
# =============================================================================

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_loss_log(log: list[tuple[int, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])
        writer.writerows(log)


def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


# =============================================================================
# Sinusoidal time embedding
# =============================================================================

def sinusoidal_embedding(t: torch.Tensor, dim: int = TIME_EMB) -> torch.Tensor:
    """
    Map scalar time t in [0, 1] to a sinusoidal embedding of shape (B, dim).

    This function is differentiable with respect to t, which is required for
    MeanFlow's JVP computation.
    """
    assert dim % 2 == 0

    half = dim // 2

    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / max(half - 1, 1)
    )

    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)

    return torch.cat(
        [
            torch.sin(args),
            torch.cos(args),
        ],
        dim=-1,
    )


# =============================================================================
# Standard Flow Matching model: x-prediction
# =============================================================================

class FlowModel(nn.Module):
    """
    Standard x-prediction flow model.

    Input:
        [z_t ; sinusoidal(t)] in R^{D + 128}

    Output:
        x_hat in R^D

    Architecture:
        5 hidden layers, hidden width 256, ReLU.
    """

    def __init__(self, data_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(data_dim + TIME_EMB, HIDDEN),
            nn.ReLU(),

            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),

            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),

            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),

            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),

            nn.Linear(HIDDEN, data_dim),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_embedding(t)
        return self.net(torch.cat([z, t_emb], dim=-1))


# =============================================================================
# MeanFlow model
# =============================================================================

class MeanFlowModel(nn.Module):
    """
    MeanFlow model.

    It predicts the average velocity u_theta(z_t, r, t).

    Input:
        [z_t ; sinusoidal(t) ; sinusoidal(h)]

    where:
        h = t - r

    Output:
        u_hat in R^D

    The forward method accepts (z, r, t) separately so that torch.func.jvp can
    compute derivatives with respect to each argument.
    """

    def __init__(self, data_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(data_dim + 2 * TIME_EMB, HIDDEN),
            nn.ReLU(),

            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),

            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),

            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),

            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),

            nn.Linear(HIDDEN, data_dim),
        )

    def forward(self, z: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = t - r

        t_emb = sinusoidal_embedding(t)
        h_emb = sinusoidal_embedding(h)

        return self.net(torch.cat([z, t_emb, h_emb], dim=-1))


# =============================================================================
# Loss functions
# =============================================================================

def compute_xpred_loss(
    model: FlowModel,
    x: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Standard flow matching with x-prediction and x-loss.

    Forward process:
        z_t = (1-t) x + t eps

    Model target:
        x
    """
    B, D = x.shape

    t = torch.rand(B, device=device)
    t = t.clamp(T_EPS, 1.0 - T_EPS)

    eps = torch.randn(B, D, device=device)

    t_col = t.unsqueeze(1)

    z_t = (1.0 - t_col) * x + t_col * eps

    x_hat = model(z_t, t)

    return ((x_hat - x) ** 2).mean()


def compute_meanflow_loss(
    model: MeanFlowModel,
    x: torch.Tensor,
    device: torch.device,
    fm_ratio: float = FM_RATIO,
) -> torch.Tensor:
    """
    MeanFlow loss.

    Linear interpolant:
        z_t = (1-t) x + t eps

    Instantaneous conditional velocity:
        v_t = eps - x

    MeanFlow identity:
        v_t = u_theta(z_t, r, t) + (t-r) * d/dt u_theta(z_t, r, t)

    Therefore:
        u_tgt = v_t - (t-r) * d/dt u_theta(z_t, r, t)

    JVP tangent for inputs (z_t, r, t):
        dz_t/dt = v_t
        dr/dt   = 0
        dt/dt   = 1

    The target is stop-gradient. Therefore, gradients are not propagated
    through the JVP term; it is used only to construct the detached target.
    """
    B, D = x.shape

    # -------------------------------------------------------------------------
    # Sample (r, t)
    # -------------------------------------------------------------------------

    t = torch.rand(B, device=device)
    t = t.clamp(T_EPS, 1.0 - T_EPS)

    use_fm = torch.rand(B, device=device) < fm_ratio

    # MeanFlow case: sample r uniformly from [0, t]
    r_mf = torch.rand(B, device=device) * t

    # FM case: r = t, so h = 0
    r = torch.where(use_fm, t.clone(), r_mf)

    h_col = (t - r).unsqueeze(1)

    # -------------------------------------------------------------------------
    # Construct z_t and v_t
    # -------------------------------------------------------------------------

    eps = torch.randn(B, D, device=device)

    t_col = t.unsqueeze(1)

    z_t = (1.0 - t_col) * x + t_col * eps

    v_t = eps - x

    # -------------------------------------------------------------------------
    # Model forward for gradient update
    # -------------------------------------------------------------------------

    u_out = model(z_t, r, t)

    # -------------------------------------------------------------------------
    # JVP for detached target
    # -------------------------------------------------------------------------

    with torch.no_grad():
        def fn(z_in: torch.Tensor, r_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
            return model(z_in, r_in, t_in)

        _, dudt = torch.func.jvp(
            fn,
            (z_t.detach(), r.detach(), t.detach()),
            (
                v_t.detach(),
                torch.zeros_like(r),
                torch.ones_like(t),
            ),
        )

    u_tgt = (v_t - h_col * dudt).detach()

    return ((u_out - u_tgt) ** 2).mean()


# =============================================================================
# Sampling
# =============================================================================

@torch.no_grad()
def euler_sample_xpred(
    model: FlowModel,
    n: int,
    dim: int,
    device: torch.device,
    n_steps: int,
) -> torch.Tensor:
    """
    Euler ODE sampling for x-prediction.

    Convert x_hat to velocity:
        v_hat = (z_t - x_hat) / t

    Then update:
        z_{t+dt} = z_t + dt * v_hat

    where dt < 0 because we integrate from t=1 to t=0.
    """
    model.eval()

    z = torch.randn(n, dim, device=device)

    ts = torch.linspace(
        1.0,
        0.0,
        n_steps + 1,
        device=device,
    )

    for i in range(n_steps):
        t_curr = ts[i]
        t_next = ts[i + 1]

        dt = t_next - t_curr

        t_batch = t_curr.expand(n)

        x_hat = model(z, t_batch)

        t_safe = t_curr.clamp(min=T_EPS)

        v_hat = (z - x_hat) / t_safe

        z = z + dt * v_hat

    model.train()

    return z


@torch.no_grad()
def meanflow_sample(
    model: MeanFlowModel,
    n: int,
    dim: int,
    device: torch.device,
    n_steps: int,
) -> torch.Tensor:
    """
    MeanFlow sampling.

    For each interval [r, t]:

        z_r = z_t - (t-r) * u_theta(z_t, r, t)

    One-step generation corresponds to:

        z_0 = z_1 - u_theta(z_1, 0, 1)
    """
    model.eval()

    z = torch.randn(n, dim, device=device)

    ts = torch.linspace(
        1.0,
        0.0,
        n_steps + 1,
        device=device,
    )

    for i in range(n_steps):
        t_curr = ts[i]
        r_next = ts[i + 1]

        h = t_curr - r_next

        t_batch = t_curr.expand(n)
        r_batch = r_next.expand(n)

        u = model(z, r_batch, t_batch)

        z = z - h * u

    model.train()

    return z


# =============================================================================
# Generic training loop
# =============================================================================

def train_model(
    model: nn.Module,
    loss_fn: Callable,
    dataset_name: str,
    device: torch.device,
    n_steps: int,
    tag: str,
) -> tuple[list[tuple[int, float]], float]:
    """
    Train one model and return:
        loss_log, elapsed_seconds
    """
    loader = get_dataloader(
        dataset_name,
        dim=DIM,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    data_iter = iter(loader)

    optimizer = Adam(model.parameters(), lr=LR)

    loss_log: list[tuple[int, float]] = []

    start_time = time.perf_counter()

    for step in range(1, n_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        batch = batch.float().to(device)

        loss = loss_fn(model, batch, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % LOG_EVERY == 0 or step == n_steps:
            loss_log.append((step, float(loss.item())))

        if step % 5_000 == 0 or step == n_steps:
            print(f"    {tag} step {step:6d}/{n_steps} loss={loss.item():.6f}")

    elapsed = time.perf_counter() - start_time

    return loss_log, elapsed


# =============================================================================
# Plotting helpers
# =============================================================================

def scatter(
    ax,
    xy: np.ndarray,
    color: str,
    title: str,
    s: float = 2,
    alpha: float = 0.4,
) -> None:
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=s,
        alpha=alpha,
        c=color,
        rasterized=True,
    )

    ax.set_title(title, fontsize=8)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)


def make_sampling_efficiency_figure(
    dataset_name: str,
    gt_2d: np.ndarray,
    samples_2d_by_steps: dict[int, np.ndarray],
    fig_path: Path,
) -> None:
    """
    One 3x3 figure:

        GT | 1 | 2
        5  | 10 | 20
        50 | 100 | 200
    """
    panels: list[tuple[str, np.ndarray, str]] = [
        ("Ground truth", gt_2d, "steelblue"),
    ]

    for steps in STEP_COUNTS_61:
        panels.append(
            (
                f"FM {steps} step{'s' if steps > 1 else ''}",
                samples_2d_by_steps[steps],
                "coral",
            )
        )

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(10.5, 10.5),
        constrained_layout=True,
    )

    axes_flat = axes.flatten()

    fig.suptitle(
        f"6.1 Sampling Efficiency – {dataset_name}, D={DIM}, x-prediction",
        fontsize=12,
    )

    for ax, (title, data, color) in zip(axes_flat, panels):
        scatter(ax, data, color, title)

    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    print(f"  Saved figure -> {fig_path}")


def make_meanflow_single_figure(
    dataset_name: str,
    steps: int,
    gt_2d: np.ndarray,
    samples_2d: np.ndarray,
    fig_path: Path,
) -> None:
    label = f"{steps} step{'s' if steps > 1 else ''}"

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8, 4),
        constrained_layout=True,
    )

    fig.suptitle(
        f"6.2 MeanFlow – {dataset_name}, D={DIM}, {label} "
        f"(FM ratio={FM_RATIO})",
        fontsize=10,
    )

    scatter(
        axes[0],
        gt_2d,
        "steelblue",
        "Ground truth",
        s=3,
        alpha=0.5,
    )

    scatter(
        axes[1],
        samples_2d,
        "mediumpurple",
        f"MeanFlow {label}",
        s=3,
        alpha=0.5,
    )

    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    print(f"  Saved figure -> {fig_path}")


def make_comparison_figure(
    dataset_name: str,
    raw_dir: Path,
    fig_dir: Path,
) -> None:
    """
    Create a side-by-side FM vs MeanFlow comparison figure.

    Layout:
        Ground Truth | FM 1 | FM 2
        FM 5        | FM 10 | FM 50
        MeanFlow 1  | MeanFlow 2 | MeanFlow 5
    """
    fm_path = raw_dir / f"6.1_{dataset_name}_D32_steps.npz"
    mf_path = raw_dir / f"6.2_{dataset_name}_D32_mf.npz"

    if not fm_path.exists() or not mf_path.exists():
        print(f"  [warning] Missing raw files for comparison: {dataset_name}")
        return

    fm = np.load(fm_path)
    mf = np.load(mf_path)

    panels = [
        ("Ground truth", fm["gt_2d"], "steelblue"),
        ("FM 1 step", fm["steps_1_2d"], "coral"),
        ("FM 2 steps", fm["steps_2_2d"], "coral"),
        ("FM 5 steps", fm["steps_5_2d"], "coral"),
        ("FM 10 steps", fm["steps_10_2d"], "coral"),
        ("FM 50 steps", fm["steps_50_2d"], "coral"),
        ("MeanFlow 1 step", mf["steps_1_2d"], "mediumpurple"),
        ("MeanFlow 2 steps", mf["steps_2_2d"], "mediumpurple"),
        ("MeanFlow 5 steps", mf["steps_5_2d"], "mediumpurple"),
    ]

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(10.5, 10.5),
        constrained_layout=True,
    )

    axes_flat = axes.flatten()

    fig.suptitle(
        f"FM vs MeanFlow Comparison – {dataset_name}, D={DIM}",
        fontsize=12,
    )

    for ax, (title, data, color) in zip(axes_flat, panels):
        scatter(ax, data, color, title)

    fig_path = fig_dir / f"6.compare_{dataset_name}_D32.png"

    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    print(f"  Saved comparison figure -> {fig_path}")


# =============================================================================
# 6.1 Sampling Efficiency
# =============================================================================

def run_61(
    device: torch.device,
    fig_dir: Path,
    raw_dir: Path,
    loss_dir: Path,
) -> list[dict]:
    """
    Train one x-prediction model per dataset at D=32.
    Evaluate sampling quality across Euler step counts.
    """
    print("\n" + "=" * 80)
    print("6.1 Sampling Efficiency")
    print("=" * 80)

    rows: list[dict] = []

    for dataset_name in DATASETS:
        print(f"\nDataset: {dataset_name}")

        ds = ToyDiffusionDataset(
            name=dataset_name,
            dim=DIM,
        )

        gt_raw = ds.data.numpy()
        gt_2d = ds.to_2d(gt_raw)

        n_samples = len(gt_raw)

        model = FlowModel(data_dim=DIM).to(device)

        model_params = count_params(model)

        print("  Training standard FM x-prediction model...")

        loss_log, elapsed = train_model(
            model=model,
            loss_fn=compute_xpred_loss,
            dataset_name=dataset_name,
            device=device,
            n_steps=N_TRAIN_FM,
            tag=f"{dataset_name} FM",
        )

        loss_path = loss_dir / f"6.1_{dataset_name}_D32_loss.csv"
        save_loss_log(loss_log, loss_path)

        samples_raw_by_steps: dict[int, np.ndarray] = {}
        samples_2d_by_steps: dict[int, np.ndarray] = {}

        for steps in STEP_COUNTS_61:
            samples_raw = euler_sample_xpred(
                model=model,
                n=n_samples,
                dim=DIM,
                device=device,
                n_steps=steps,
            ).cpu().numpy()

            samples_2d = ds.to_2d(samples_raw)

            samples_raw_by_steps[steps] = samples_raw
            samples_2d_by_steps[steps] = samples_2d

            print(f"  Sampled standard FM with {steps} step{'s' if steps > 1 else ''}")

        raw_arrays = {
            "gt_raw": gt_raw,
            "gt_2d": gt_2d,
        }

        for steps in STEP_COUNTS_61:
            raw_arrays[f"steps_{steps}_raw"] = samples_raw_by_steps[steps]
            raw_arrays[f"steps_{steps}_2d"] = samples_2d_by_steps[steps]

        raw_path = raw_dir / f"6.1_{dataset_name}_D32_steps.npz"

        save_npz(
            raw_path,
            **raw_arrays,
        )

        fig_path = fig_dir / f"6.1_{dataset_name}_D32_steps.png"

        make_sampling_efficiency_figure(
            dataset_name=dataset_name,
            gt_2d=gt_2d,
            samples_2d_by_steps=samples_2d_by_steps,
            fig_path=fig_path,
        )

        rows.append(
            {
                "section": "6.1",
                "dataset": dataset_name,
                "dim": DIM,
                "config": "standard_fm_xpred_xloss",
                "model_type": "FlowModel",
                "prediction_type": "x",
                "train_steps": N_TRAIN_FM,
                "sample_steps": "1/2/5/10/20/50/100/200",
                "uses_jvp": False,
                "fm_ratio": "",
                "model_params": model_params,
                "relative_step_cost": 1.0,
                "elapsed_seconds": elapsed,
                "final_loss": loss_log[-1][1],
                "figure_path": str(fig_path),
                "raw_path": str(raw_path),
                "loss_path": str(loss_path),
            }
        )

    return rows


# =============================================================================
# 6.2 MeanFlow
# =============================================================================

def run_62(
    device: torch.device,
    fig_dir: Path,
    raw_dir: Path,
    loss_dir: Path,
) -> list[dict]:
    """
    Train one MeanFlow model per dataset at D=32.
    Evaluate 1, 2, and 5 step MeanFlow generation.
    """
    print("\n" + "=" * 80)
    print("6.2 MeanFlow")
    print("=" * 80)

    rows: list[dict] = []

    for dataset_name in DATASETS:
        print(f"\nDataset: {dataset_name}")

        ds = ToyDiffusionDataset(
            name=dataset_name,
            dim=DIM,
        )

        gt_raw = ds.data.numpy()
        gt_2d = ds.to_2d(gt_raw)

        n_samples = len(gt_raw)

        model = MeanFlowModel(data_dim=DIM).to(device)

        model_params = count_params(model)

        print("  Training MeanFlow model...")

        loss_log, elapsed = train_model(
            model=model,
            loss_fn=compute_meanflow_loss,
            dataset_name=dataset_name,
            device=device,
            n_steps=N_TRAIN_MF,
            tag=f"{dataset_name} MeanFlow",
        )

        loss_path = loss_dir / f"6.2_{dataset_name}_D32_mf_loss.csv"
        save_loss_log(loss_log, loss_path)

        samples_raw_by_steps: dict[int, np.ndarray] = {}
        samples_2d_by_steps: dict[int, np.ndarray] = {}

        for steps in STEP_COUNTS_62:
            samples_raw = meanflow_sample(
                model=model,
                n=n_samples,
                dim=DIM,
                device=device,
                n_steps=steps,
            ).cpu().numpy()

            samples_2d = ds.to_2d(samples_raw)

            samples_raw_by_steps[steps] = samples_raw
            samples_2d_by_steps[steps] = samples_2d

            print(f"  Sampled MeanFlow with {steps} step{'s' if steps > 1 else ''}")

        raw_arrays = {
            "gt_raw": gt_raw,
            "gt_2d": gt_2d,
        }

        for steps in STEP_COUNTS_62:
            raw_arrays[f"steps_{steps}_raw"] = samples_raw_by_steps[steps]
            raw_arrays[f"steps_{steps}_2d"] = samples_2d_by_steps[steps]

        raw_path = raw_dir / f"6.2_{dataset_name}_D32_mf.npz"

        save_npz(
            raw_path,
            **raw_arrays,
        )

        # Produce 9 required figures:
        # 3 datasets x 3 MeanFlow step counts.
        for steps in STEP_COUNTS_62:
            fig_path = fig_dir / f"6.2_{dataset_name}_{steps}step.png"

            make_meanflow_single_figure(
                dataset_name=dataset_name,
                steps=steps,
                gt_2d=gt_2d,
                samples_2d=samples_2d_by_steps[steps],
                fig_path=fig_path,
            )

        rows.append(
            {
                "section": "6.2",
                "dataset": dataset_name,
                "dim": DIM,
                "config": f"meanflow_fmratio_{FM_RATIO}",
                "model_type": "MeanFlowModel",
                "prediction_type": "mean_velocity",
                "train_steps": N_TRAIN_MF,
                "sample_steps": "1/2/5",
                "uses_jvp": True,
                "fm_ratio": FM_RATIO,
                "model_params": model_params,
                "relative_step_cost": 2.0,
                "elapsed_seconds": elapsed,
                "final_loss": loss_log[-1][1],
                "figure_path": f"{fig_dir}/6.2_{dataset_name}_{{1,2,5}}step.png",
                "raw_path": str(raw_path),
                "loss_path": str(loss_path),
            }
        )

    return rows


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    root = Path(__file__).resolve().parent

    out_dir = root / "part6_results"

    fig_dir = out_dir / "figures"
    raw_dir = out_dir / "raw"
    loss_dir = out_dir / "losses"

    fig_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    loss_dir.mkdir(parents=True, exist_ok=True)

    rows_61 = run_61(
        device=device,
        fig_dir=fig_dir,
        raw_dir=raw_dir,
        loss_dir=loss_dir,
    )

    rows_62 = run_62(
        device=device,
        fig_dir=fig_dir,
        raw_dir=raw_dir,
        loss_dir=loss_dir,
    )

    print("\n" + "=" * 80)
    print("Creating FM vs MeanFlow comparison figures")
    print("=" * 80)

    for dataset_name in DATASETS:
        make_comparison_figure(
            dataset_name=dataset_name,
            raw_dir=raw_dir,
            fig_dir=fig_dir,
        )

    # -------------------------------------------------------------------------
    # Summary CSV
    # -------------------------------------------------------------------------

    summary_path = out_dir / "summary.csv"

    all_rows = rows_61 + rows_62

    fieldnames = [
        "section",
        "dataset",
        "dim",
        "config",
        "model_type",
        "prediction_type",
        "train_steps",
        "sample_steps",
        "uses_jvp",
        "fm_ratio",
        "model_params",
        "relative_step_cost",
        "elapsed_seconds",
        "final_loss",
        "figure_path",
        "raw_path",
        "loss_path",
    ]

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in all_rows:
            writer.writerow(row)

    print("\n" + "=" * 80)
    print("All done.")
    print(f"Results saved in: {out_dir}")
    print(f"Figures: {fig_dir}")
    print(f"Raw arrays: {raw_dir}")
    print(f"Loss logs: {loss_dir}")
    print(f"Summary: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()