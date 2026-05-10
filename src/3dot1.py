import numpy as np
import matplotlib.pyplot as plt
from dataloader import ToyDiffusionDataset

datasets = ["swiss_roll", "gaussians", "circles"]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for col, name in enumerate(datasets):
    # 原始 2D 数据
    ds_2d = ToyDiffusionDataset(name=name, dim=2)
    data_2d = ds_2d.data.numpy()

    # 32 维数据投影回 2D
    ds_32 = ToyDiffusionDataset(name=name, dim=32)
    data_32 = ds_32.data.numpy()
    data_back = ds_32.to_2d(data_32)   # shape: (N, 2)

    # 上排：原始 2D
    axes[0, col].scatter(data_2d[:, 0], data_2d[:, 1], s=2, alpha=0.5)
    axes[0, col].set_title(f"{name} — original 2D")

    # 下排：32D 投影回 2D
    axes[1, col].scatter(data_back[:, 0], data_back[:, 1], s=2, alpha=0.5)
    axes[1, col].set_title(f"{name} — back-projected (32D→2D)")

plt.tight_layout()
plt.savefig("part1_visualization.png", dpi=150)
plt.show()