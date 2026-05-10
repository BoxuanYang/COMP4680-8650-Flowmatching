"""
5.py
Part 3 – Can We Rescue v-Prediction?
====================================

This script runs the Part 3 rescue experiments for v-prediction.

Task:
    From Part 2, v-prediction fails at D = 32 with the default model and
    training setup. We test whether this failure is fundamental or can be
    overcome.

Dataset:
    swiss_roll, D = 32

Experiments:
    A. Default baselines
        A1. default v-pred + v-loss, ambient D=32
        A2. default x-pred + x-loss, ambient D=32

        The script first tries to load these from:
            src/part4_results/raw/

        If the files are not found, it trains them from scratch.

    B. Scaled ambient-space v-prediction
        v-pred + v-loss in ambient D=32
        hidden_dim = 512
        training steps = 50,000

        This tests whether moderate model/compute scaling improves
        ambient-space v-prediction.

    C. RAE-inspired latent flow matching
        1. Train an MLP autoencoder on swiss_roll D=32.
        2. Encode data into a low-dimensional latent space.
        3. Standardize latent codes.
        4. Train v-pred + v-loss flow matching in latent space.
        5. Sample in latent space, unstandardize, decode back to D=32.
        6. Project decoded samples to 2D for visualization.

        This is a toy analogue of RAE-style representation-space generation:
        the flow model operates in a learned representation space rather than
        the raw ambient space.

Outputs:
    part5_results/
        figures/
            A_baseline_swiss_roll_D32.png
            B_scaled_model_swiss_roll_D32.png
            C_latent_fm_swiss_roll_D32.png
            summary.png

        raw/
            ground_truth.npz
            A_default_vpred_vloss.npz
            A_default_xpred_xloss.npz
            B_scaled_vpred_vloss.npz
            C_latent_k2_vpred_vloss.npz
            C_latent_k4_vpred_vloss.npz

        losses/
            A_default_vpred_vloss_loss.csv       if trained from scratch
            A_default_xpred_xloss_loss.csv       if trained from scratch
            B_scaled_vpred_vloss_loss.csv
            C_latent_k2_ae_loss.csv
            C_latent_k2_fm_loss.csv
            C_latent_k4_ae_loss.csv
            C_latent_k4_fm_loss.csv

        summary.csv

Hyperparameters:
    Default FM model:
        5 hidden layers, hidden_dim = 256, ReLU
        128-d sinusoidal time embedding
        Adam, lr = 1e-3
        batch size = 1024
        FM steps = 25,000
        Euler sampling steps = 50

    Scaled FM model:
        hidden_dim = 512
        FM steps = 50,000

    AE:
        Encoder: Linear(32 -> 256) + ReLU + Linear(256 -> latent_dim)
        Decoder: Linear(latent_dim -> 256) + ReLU + Linear(256 -> 32)
        MSE reconstruction loss
        Adam, lr = 1e-3
        AE steps = 20,000
"""

import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# Paths and imports
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR

sys.path.insert(0, str(SCRIPT_DIR))
from dataloader import ToyDiffusionDataset


# =============================================================================
# Shared hyperparameters
# =============================================================================

DATASET_NAME = "swiss_roll"
DIM = 32

T_EPS = 1e-3

TIME_EMB_DIM = 128
HIDDEN_DIM_DEFAULT = 256
HIDDEN_DIM_SCALED = 512

N_FM_STEPS_DEFAULT = 25_000
N_FM_STEPS_SCALED = 50_000

BATCH_SIZE = 1024
LR = 1e-3

N_SAMPLE_STEPS_DEFAULT = 50

N_AE_STEPS = 20_000
AE_HIDDEN_DIM = 256

LATENT_DIMS = [2, 4]

LOSS_LOG_INTERVAL = 500


# =============================================================================
# Utility functions
# =============================================================================

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def write_loss_csv(loss_log: list[tuple[int, float]], path: Path, value_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", value_name])
        for step, value in loss_log:
            writer.writerow([step, value])


def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def compute_relative_cost(
    flow_params: int,
    fm_steps: int,
    sample_steps: int,
    ref_flow_params: int,
    ref_fm_steps: int,
    ref_sample_steps: int,
    ae_params: int = 0,
    ae_steps: int = 0,
) -> tuple[float, float]:
    """
    Approximate compute cost.

    Train cost:
        flow_params * fm_steps + ae_params * ae_steps

    Sampling cost:
        flow_params * sample_steps

    The AE decode cost during sampling is ignored because it is one decoder
    pass per sample batch, while the flow sampler uses many ODE evaluations.
    """
    ref_train = ref_flow_params * ref_fm_steps
    ref_sample = ref_flow_params * ref_sample_steps

    train_cost = flow_params * fm_steps + ae_params * ae_steps
    sample_cost = flow_params * sample_steps

    return train_cost / ref_train, sample_cost / ref_sample


# =============================================================================
# Sinusoidal time embedding
# =============================================================================

def sinusoidal_embedding(t: torch.Tensor, dim: int = TIME_EMB_DIM) -> torch.Tensor:
    """
    Map scalar time t ∈ [0,1] to a sinusoidal embedding of shape (B, dim).

    Frequencies:
        omega_i = exp(- i * ln(10000) / (k - 1))
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

    return torch.cat(
        [
            torch.sin(args),
            torch.cos(args),
        ],
        dim=-1,
    )


# =============================================================================
# Flow matching model
# =============================================================================

class FlowModel(nn.Module):
    """
    Assignment-specified MLP, with configurable hidden width.

    Input:
        [z_t ; e_t] ∈ R^{D + 128}

    Architecture:
        5 hidden layers, each Linear -> ReLU
        final Linear output layer

    The raw output is interpreted according to pred_type:
        pred_type == "x": raw output is x_hat
        pred_type == "v": raw output is v_hat
    """

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int = HIDDEN_DIM_DEFAULT,
        time_emb_dim: int = TIME_EMB_DIM,
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
        t_emb = sinusoidal_embedding(t, self.time_emb_dim)
        return self.net(torch.cat([z, t_emb], dim=-1))


# =============================================================================
# Flow matching loss and sampler
# =============================================================================

def compute_fm_loss(
    model: FlowModel,
    x: torch.Tensor,
    pred_type: str,
    loss_type: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Linear interpolant:
        z_t = (1 - t) x + t eps

    True velocity:
        v = eps - x

    Conversion:
        x_hat -> v_hat:
            v_hat = (z_t - x_hat) / t

        v_hat -> x_hat:
            x_hat = z_t - t v_hat
    """
    assert pred_type in ["x", "v"]
    assert loss_type in ["x", "v"]

    B, D = x.shape

    t = torch.rand(B, device=device)
    t = t.clamp(T_EPS, 1.0 - T_EPS)
    t_col = t.unsqueeze(1)

    eps = torch.randn(B, D, device=device)

    z_t = (1.0 - t_col) * x + t_col * eps
    v = eps - x

    raw = model(z_t, t)

    if pred_type == "x":
        x_hat = raw
        v_hat = (z_t - x_hat) / t_col
    else:
        v_hat = raw
        x_hat = z_t - t_col * v_hat

    if loss_type == "x":
        return ((x_hat - x) ** 2).mean()
    else:
        return ((v_hat - v) ** 2).mean()


@torch.no_grad()
def euler_sample(
    model: FlowModel,
    n_samples: int,
    dim: int,
    pred_type: str,
    device: torch.device,
    n_steps: int = N_SAMPLE_STEPS_DEFAULT,
) -> torch.Tensor:
    """
    Generate samples by integrating from t=1 to t=0.

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

        dt = t_next - t_curr

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


def train_fm(
    data_tensor: torch.Tensor,
    pred_type: str,
    loss_type: str,
    hidden_dim: int,
    n_steps: int,
    device: torch.device,
    label: str,
) -> tuple[FlowModel, list[tuple[int, float]]]:
    """
    Train one flow matching model.
    """
    data_dim = data_tensor.shape[1]

    dataset = TensorDataset(data_tensor)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    data_iter = iter(loader)

    model = FlowModel(
        data_dim=data_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=LR)

    loss_log: list[tuple[int, float]] = []

    print(
        f"\nTraining FM [{label}] "
        f"pred={pred_type}, loss={loss_type}, "
        f"dim={data_dim}, hidden={hidden_dim}, steps={n_steps}"
    )

    for step in range(1, n_steps + 1):
        try:
            (batch,) = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            (batch,) = next(data_iter)

        batch = batch.float().to(device)

        loss = compute_fm_loss(
            model=model,
            x=batch,
            pred_type=pred_type,
            loss_type=loss_type,
            device=device,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % LOSS_LOG_INTERVAL == 0 or step == n_steps:
            loss_log.append((step, float(loss.item())))

        if step % 5_000 == 0 or step == n_steps:
            print(f"  step {step:6d}/{n_steps}  fm_loss={loss.item():.6f}")

    return model, loss_log


# =============================================================================
# Autoencoder for latent-space rescue
# =============================================================================

class MLP_AE(nn.Module):
    """
    Lightweight MLP autoencoder for swiss_roll D=32.

    This is an RAE-inspired toy analogue:
        - RAE uses a frozen pretrained image representation encoder.
        - Here, we learn a small encoder for the synthetic swiss-roll data.

    The key mechanism is representation-space generation:
        instead of training v-prediction in raw ambient D=32 space,
        we train v-prediction in latent dimension k << D.
    """

    def __init__(
        self,
        input_dim: int = DIM,
        latent_dim: int = 2,
        hidden_dim: int = AE_HIDDEN_DIM,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def train_ae(
    data_tensor: torch.Tensor,
    latent_dim: int,
    device: torch.device,
) -> tuple[MLP_AE, list[tuple[int, float]]]:
    """
    Train the MLP autoencoder.
    """
    input_dim = data_tensor.shape[1]

    dataset = TensorDataset(data_tensor)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    data_iter = iter(loader)

    ae = MLP_AE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=AE_HIDDEN_DIM,
    ).to(device)

    optimizer = Adam(ae.parameters(), lr=LR)

    loss_log: list[tuple[int, float]] = []

    print(
        f"\nTraining AE "
        f"D={input_dim} -> latent={latent_dim}, steps={N_AE_STEPS}"
    )

    for step in range(1, N_AE_STEPS + 1):
        try:
            (batch,) = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            (batch,) = next(data_iter)

        batch = batch.float().to(device)

        recon = ae(batch)
        loss = ((recon - batch) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % LOSS_LOG_INTERVAL == 0 or step == N_AE_STEPS:
            loss_log.append((step, float(loss.item())))

        if step % 5_000 == 0 or step == N_AE_STEPS:
            print(f"  step {step:6d}/{N_AE_STEPS}  recon_loss={loss.item():.6f}")

    return ae, loss_log


# =============================================================================
# Loading Part 2 / Part 4 raw baseline results
# =============================================================================

def find_part4_raw_dir() -> Path | None:
    """
    Try to locate part4_results/raw.

    Given the user's current layout:
        /workspace/COMP4680-8650-Flowmatching/src/
            4.py
            5.py
            dataloader.py
            part4_results/

    the expected path is:
        SCRIPT_DIR / "part4_results" / "raw"
    """
    candidates = [
        SCRIPT_DIR / "part4_results" / "raw",
        SCRIPT_DIR.parent / "part4_results" / "raw",
    ]

    for p in candidates:
        if p.exists():
            return p

    return None


def try_load_part4_raw(
    pred_type: str,
    loss_type: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load Part 2 raw result.

    Expected filename:
        swiss_roll_D32_xpred_xloss.npz
        swiss_roll_D32_vpred_vloss.npz
    """
    part4_raw_dir = find_part4_raw_dir()

    if part4_raw_dir is None:
        print("  Part 4 raw directory not found.")
        return None

    tag = f"{DATASET_NAME}_D{DIM}_{pred_type}pred_{loss_type}loss"
    path = part4_raw_dir / f"{tag}.npz"

    if not path.exists():
        print(f"  Part 4 raw file not found: {path}")
        return None

    print(f"  Loaded Part 4 raw baseline: {path}")

    data = np.load(path)

    samples_raw = data["samples_raw"]
    samples_2d = data["samples_2d"]

    return samples_raw, samples_2d


# =============================================================================
# Plotting
# =============================================================================

def scatter(
    ax,
    data: np.ndarray,
    color: str,
    title: str,
) -> None:
    ax.scatter(
        data[:, 0],
        data[:, 1],
        s=2,
        alpha=0.4,
        c=color,
        rasterized=True,
    )

    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)


def save_panel_figure(
    panels: list[tuple[str, np.ndarray, str]],
    title: str,
    path: Path,
    n_rows: int = 1,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    n_panels = len(panels)

    if n_rows == 1:
        n_cols = n_panels
    else:
        n_cols = math.ceil(n_panels / n_rows)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 5 * n_rows),
        squeeze=False,
    )

    axes_flat = axes.flatten()

    fig.suptitle(title, fontsize=12)

    for i, (panel_title, data, color) in enumerate(panels):
        scatter(
            axes_flat[i],
            data,
            color=color,
            title=panel_title,
        )

    for j in range(len(panels), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

    print(f"Saved figure -> {path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = SCRIPT_DIR / "part5_results"
    fig_dir = out_dir / "figures"
    raw_dir = out_dir / "raw"
    loss_dir = out_dir / "losses"

    fig_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    loss_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []

    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------

    ds = ToyDiffusionDataset(
        name=DATASET_NAME,
        dim=DIM,
    )

    gt_raw = ds.data.numpy()
    gt_2d = ds.to_2d(gt_raw)
    data_tensor = ds.data.float()

    n_samples = len(gt_raw)

    save_npz(
        raw_dir / "ground_truth.npz",
        gt_raw=gt_raw,
        gt_2d=gt_2d,
    )

    # Reference compute: default x-pred setup
    ref_model = FlowModel(
        data_dim=DIM,
        hidden_dim=HIDDEN_DIM_DEFAULT,
    )
    ref_flow_params = count_params(ref_model)
    ref_fm_steps = N_FM_STEPS_DEFAULT
    ref_sample_steps = N_SAMPLE_STEPS_DEFAULT

    print(f"Reference FlowModel params: {ref_flow_params}")

    # -------------------------------------------------------------------------
    # Experiment A: Default baselines
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("Experiment A: Default baselines")
    print("=" * 80)

    baseline_results: dict[str, dict[str, object]] = {}

    baseline_configs = [
        {
            "name": "A_default_vpred_vloss",
            "pred_type": "v",
            "loss_type": "v",
            "color": "coral",
            "title": "A: default v-pred + v-loss",
        },
        {
            "name": "A_default_xpred_xloss",
            "pred_type": "x",
            "loss_type": "x",
            "color": "mediumseagreen",
            "title": "A: default x-pred + x-loss",
        },
    ]

    for cfg in baseline_configs:
        name = cfg["name"]
        pred_type = cfg["pred_type"]
        loss_type = cfg["loss_type"]

        loaded = try_load_part4_raw(
            pred_type=pred_type,
            loss_type=loss_type,
        )

        loss_path = ""
        trained_from_scratch = False

        if loaded is not None:
            samples_raw, samples_2d = loaded
            final_loss = np.nan

        else:
            print(f"  Training baseline from scratch: {name}")

            model, loss_log = train_fm(
                data_tensor=data_tensor,
                pred_type=pred_type,
                loss_type=loss_type,
                hidden_dim=HIDDEN_DIM_DEFAULT,
                n_steps=N_FM_STEPS_DEFAULT,
                device=device,
                label=name,
            )

            loss_path_obj = loss_dir / f"{name}_loss.csv"
            write_loss_csv(
                loss_log=loss_log,
                path=loss_path_obj,
                value_name="fm_loss",
            )

            loss_path = str(loss_path_obj)
            final_loss = float(loss_log[-1][1])
            trained_from_scratch = True

            samples_raw = euler_sample(
                model=model,
                n_samples=n_samples,
                dim=DIM,
                pred_type=pred_type,
                device=device,
                n_steps=N_SAMPLE_STEPS_DEFAULT,
            ).cpu().numpy()

            samples_2d = ds.to_2d(samples_raw)

        raw_path = raw_dir / f"{name}.npz"

        save_npz(
            raw_path,
            gt_raw=gt_raw,
            gt_2d=gt_2d,
            samples_raw=samples_raw,
            samples_2d=samples_2d,
            pred_type=np.array(pred_type),
            loss_type=np.array(loss_type),
            trained_from_scratch=np.array(trained_from_scratch),
        )

        flow_params = ref_flow_params
        rel_train, rel_sample = compute_relative_cost(
            flow_params=flow_params,
            fm_steps=N_FM_STEPS_DEFAULT,
            sample_steps=N_SAMPLE_STEPS_DEFAULT,
            ref_flow_params=ref_flow_params,
            ref_fm_steps=ref_fm_steps,
            ref_sample_steps=ref_sample_steps,
        )

        summary_rows.append(
            {
                "name": name,
                "method": "baseline_ambient",
                "dataset": DATASET_NAME,
                "dim": DIM,
                "space": "ambient",
                "latent_dim": "",
                "pred_type": pred_type,
                "loss_type": loss_type,
                "hidden_dim": HIDDEN_DIM_DEFAULT,
                "fm_steps": N_FM_STEPS_DEFAULT,
                "sample_steps": N_SAMPLE_STEPS_DEFAULT,
                "ae_steps": 0,
                "flow_params": flow_params,
                "ae_params": 0,
                "final_loss": final_loss,
                "relative_train_compute": rel_train,
                "relative_sample_compute": rel_sample,
                "raw_path": str(raw_path),
                "loss_path": loss_path,
            }
        )

        baseline_results[name] = {
            "samples_raw": samples_raw,
            "samples_2d": samples_2d,
            "color": cfg["color"],
            "title": cfg["title"],
        }

    # Baseline figure
    baseline_panels = [
        ("Ground truth", gt_2d, "steelblue"),
        (
            "Default v-pred + v-loss\nambient D=32",
            baseline_results["A_default_vpred_vloss"]["samples_2d"],
            "coral",
        ),
        (
            "Default x-pred + x-loss\nambient D=32",
            baseline_results["A_default_xpred_xloss"]["samples_2d"],
            "mediumseagreen",
        ),
    ]

    save_panel_figure(
        panels=baseline_panels,
        title=f"Experiment A – Default Baselines ({DATASET_NAME}, D={DIM})",
        path=fig_dir / "A_baseline_swiss_roll_D32.png",
        n_rows=1,
    )

    # -------------------------------------------------------------------------
    # Experiment B: Scaled ambient-space v-prediction
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("Experiment B: Scaled ambient-space v-prediction")
    print("=" * 80)

    name_b = "B_scaled_vpred_vloss"

    model_b, loss_log_b = train_fm(
        data_tensor=data_tensor,
        pred_type="v",
        loss_type="v",
        hidden_dim=HIDDEN_DIM_SCALED,
        n_steps=N_FM_STEPS_SCALED,
        device=device,
        label=name_b,
    )

    loss_path_b = loss_dir / f"{name_b}_loss.csv"
    write_loss_csv(
        loss_log=loss_log_b,
        path=loss_path_b,
        value_name="fm_loss",
    )

    samples_raw_b = euler_sample(
        model=model_b,
        n_samples=n_samples,
        dim=DIM,
        pred_type="v",
        device=device,
        n_steps=N_SAMPLE_STEPS_DEFAULT,
    ).cpu().numpy()

    samples_2d_b = ds.to_2d(samples_raw_b)

    raw_path_b = raw_dir / f"{name_b}.npz"

    save_npz(
        raw_path_b,
        gt_raw=gt_raw,
        gt_2d=gt_2d,
        samples_raw=samples_raw_b,
        samples_2d=samples_2d_b,
        pred_type=np.array("v"),
        loss_type=np.array("v"),
        hidden_dim=np.array(HIDDEN_DIM_SCALED),
        fm_steps=np.array(N_FM_STEPS_SCALED),
    )

    flow_params_b = count_params(model_b)

    rel_train_b, rel_sample_b = compute_relative_cost(
        flow_params=flow_params_b,
        fm_steps=N_FM_STEPS_SCALED,
        sample_steps=N_SAMPLE_STEPS_DEFAULT,
        ref_flow_params=ref_flow_params,
        ref_fm_steps=ref_fm_steps,
        ref_sample_steps=ref_sample_steps,
    )

    summary_rows.append(
        {
            "name": name_b,
            "method": "scaled_ambient_vpred",
            "dataset": DATASET_NAME,
            "dim": DIM,
            "space": "ambient",
            "latent_dim": "",
            "pred_type": "v",
            "loss_type": "v",
            "hidden_dim": HIDDEN_DIM_SCALED,
            "fm_steps": N_FM_STEPS_SCALED,
            "sample_steps": N_SAMPLE_STEPS_DEFAULT,
            "ae_steps": 0,
            "flow_params": flow_params_b,
            "ae_params": 0,
            "final_loss": float(loss_log_b[-1][1]),
            "relative_train_compute": rel_train_b,
            "relative_sample_compute": rel_sample_b,
            "raw_path": str(raw_path_b),
            "loss_path": str(loss_path_b),
        }
    )

    scaled_panels = [
        ("Ground truth", gt_2d, "steelblue"),
        (
            "Scaled ambient v-pred + v-loss\nhidden=512, 50K FM steps",
            samples_2d_b,
            "tomato",
        ),
    ]

    save_panel_figure(
        panels=scaled_panels,
        title=f"Experiment B – Scaled Ambient v-Prediction ({DATASET_NAME}, D={DIM})",
        path=fig_dir / "B_scaled_model_swiss_roll_D32.png",
        n_rows=1,
    )

    # -------------------------------------------------------------------------
    # Experiment C: RAE-inspired latent flow matching
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("Experiment C: RAE-inspired latent flow matching")
    print("=" * 80)

    latent_results: dict[int, dict[str, object]] = {}

    for latent_dim in LATENT_DIMS:
        print("\n" + "-" * 80)
        print(f"Latent dimension k={latent_dim}")
        print("-" * 80)

        name_c = f"C_latent_k{latent_dim}_vpred_vloss"

        # Train AE
        ae, ae_loss_log = train_ae(
            data_tensor=data_tensor,
            latent_dim=latent_dim,
            device=device,
        )

        ae_loss_path = loss_dir / f"C_latent_k{latent_dim}_ae_loss.csv"
        write_loss_csv(
            loss_log=ae_loss_log,
            path=ae_loss_path,
            value_name="recon_loss",
        )

        ae.eval()

        # Encode full dataset
        with torch.no_grad():
            latent_data = ae.encode(data_tensor.to(device)).cpu()

        # Standardize latent codes before FM.
        # This makes the latent distribution closer to the N(0,I)-based FM setup.
        latent_mean = latent_data.mean(dim=0, keepdim=True)
        latent_std = latent_data.std(dim=0, keepdim=True).clamp(min=1e-6)

        latent_data_std = (latent_data - latent_mean) / latent_std

        # Train latent FM with v-pred + v-loss
        model_c, fm_loss_log = train_fm(
            data_tensor=latent_data_std,
            pred_type="v",
            loss_type="v",
            hidden_dim=HIDDEN_DIM_DEFAULT,
            n_steps=N_FM_STEPS_DEFAULT,
            device=device,
            label=name_c,
        )

        fm_loss_path = loss_dir / f"C_latent_k{latent_dim}_fm_loss.csv"
        write_loss_csv(
            loss_log=fm_loss_log,
            path=fm_loss_path,
            value_name="fm_loss",
        )

        # Sample standardized latent, unstandardize, decode
        with torch.no_grad():
            latent_samples_std = euler_sample(
                model=model_c,
                n_samples=n_samples,
                dim=latent_dim,
                pred_type="v",
                device=device,
                n_steps=N_SAMPLE_STEPS_DEFAULT,
            ).cpu()

            latent_samples = latent_samples_std * latent_std + latent_mean

            decoded_raw = ae.decode(latent_samples.to(device)).cpu().numpy()

        decoded_2d = ds.to_2d(decoded_raw)

        raw_path_c = raw_dir / f"{name_c}.npz"

        save_npz(
            raw_path_c,
            gt_raw=gt_raw,
            gt_2d=gt_2d,
            latent_data=latent_data.numpy(),
            latent_data_std=latent_data_std.numpy(),
            latent_mean=latent_mean.numpy(),
            latent_std=latent_std.numpy(),
            latent_samples_std=latent_samples_std.numpy(),
            latent_samples=latent_samples.numpy(),
            decoded_raw=decoded_raw,
            samples_raw=decoded_raw,
            samples_2d=decoded_2d,
            latent_dim=np.array(latent_dim),
            pred_type=np.array("v"),
            loss_type=np.array("v"),
        )

        flow_params_c = count_params(model_c)
        ae_params_c = count_params(ae)

        rel_train_c, rel_sample_c = compute_relative_cost(
            flow_params=flow_params_c,
            fm_steps=N_FM_STEPS_DEFAULT,
            sample_steps=N_SAMPLE_STEPS_DEFAULT,
            ae_params=ae_params_c,
            ae_steps=N_AE_STEPS,
            ref_flow_params=ref_flow_params,
            ref_fm_steps=ref_fm_steps,
            ref_sample_steps=ref_sample_steps,
        )

        summary_rows.append(
            {
                "name": name_c,
                "method": "latent_vpred",
                "dataset": DATASET_NAME,
                "dim": DIM,
                "space": "latent",
                "latent_dim": latent_dim,
                "pred_type": "v",
                "loss_type": "v",
                "hidden_dim": HIDDEN_DIM_DEFAULT,
                "fm_steps": N_FM_STEPS_DEFAULT,
                "sample_steps": N_SAMPLE_STEPS_DEFAULT,
                "ae_steps": N_AE_STEPS,
                "flow_params": flow_params_c,
                "ae_params": ae_params_c,
                "final_loss": float(fm_loss_log[-1][1]),
                "relative_train_compute": rel_train_c,
                "relative_sample_compute": rel_sample_c,
                "raw_path": str(raw_path_c),
                "loss_path": str(fm_loss_path),
                "ae_loss_path": str(ae_loss_path),
            }
        )

        latent_results[latent_dim] = {
            "samples_2d": decoded_2d,
            "raw_path": raw_path_c,
            "color": "mediumpurple",
            "title": f"Latent FM k={latent_dim}\nAE + v-pred + v-loss",
        }

    latent_panels = [
        ("Ground truth", gt_2d, "steelblue"),
    ]

    for latent_dim in LATENT_DIMS:
        latent_panels.append(
            (
                f"AE latent k={latent_dim}\nlatent v-pred + v-loss",
                latent_results[latent_dim]["samples_2d"],
                "mediumpurple",
            )
        )

    save_panel_figure(
        panels=latent_panels,
        title=f"Experiment C – RAE-inspired Latent Flow Matching ({DATASET_NAME}, D={DIM})",
        path=fig_dir / "C_latent_fm_swiss_roll_D32.png",
        n_rows=1,
    )

    # -------------------------------------------------------------------------
    # Summary figure
    # -------------------------------------------------------------------------

    summary_panels = [
        ("Ground truth", gt_2d, "steelblue"),
        (
            "A: default v-pred + v-loss\nambient D=32",
            baseline_results["A_default_vpred_vloss"]["samples_2d"],
            "coral",
        ),
        (
            "A: default x-pred + x-loss\nambient D=32",
            baseline_results["A_default_xpred_xloss"]["samples_2d"],
            "mediumseagreen",
        ),
        (
            "B: scaled ambient v-pred\nhidden=512, 50K steps",
            samples_2d_b,
            "tomato",
        ),
    ]

    for latent_dim in LATENT_DIMS:
        summary_panels.append(
            (
                f"C: latent FM k={latent_dim}\nAE + v-pred",
                latent_results[latent_dim]["samples_2d"],
                "mediumpurple",
            )
        )

    save_panel_figure(
        panels=summary_panels,
        title=f"Part 3 Summary – Can We Rescue v-Prediction? ({DATASET_NAME}, D={DIM})",
        path=fig_dir / "summary.png",
        n_rows=1,
    )

    # -------------------------------------------------------------------------
    # Save summary.csv
    # -------------------------------------------------------------------------

    summary_path = out_dir / "summary.csv"

    fieldnames = [
        "name",
        "method",
        "dataset",
        "dim",
        "space",
        "latent_dim",
        "pred_type",
        "loss_type",
        "hidden_dim",
        "fm_steps",
        "sample_steps",
        "ae_steps",
        "flow_params",
        "ae_params",
        "final_loss",
        "relative_train_compute",
        "relative_sample_compute",
        "raw_path",
        "loss_path",
        "ae_loss_path",
    ]

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in summary_rows:
            writer.writerow(row)

    print("\n" + "=" * 80)
    print("Part 3 complete.")
    print(f"Results saved to: {out_dir}")
    print(f"Figures: {fig_dir}")
    print(f"Raw arrays: {raw_dir}")
    print(f"Loss logs: {loss_dir}")
    print(f"Summary CSV: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()