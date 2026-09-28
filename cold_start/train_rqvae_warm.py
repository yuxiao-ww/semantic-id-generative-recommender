"""
S7b-1: 只在 warm item 上重训 RQ-VAE

和 S3 的 train_rqvae.py 唯一区别: 训练数据排除 cold item。
这样 RQ-VAE 从未见过 cold item，后续用它编码 cold item 才是真正的
"语义泛化到新item"，构成严谨的冷启动实验。

产出: checkpoints/rqvae_warm.pt
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from semantic_id.rqvae import RQVAE      # 复用 S3 的模型结构

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def eval_metrics(model, x, device, n_layers, num_codes):
    """复用 S3 的验收指标: 重建余弦 + 每层码本使用率。"""
    model.eval()
    recon, indices, _ = model(x.to(device))
    recon_n = torch.nn.functional.normalize(recon, dim=1)
    orig_n = torch.nn.functional.normalize(x.to(device), dim=1)
    cos = (recon_n * orig_n).sum(1).mean().item()
    usage = [indices[:, l].unique().numel() for l in range(n_layers)]
    return cos, usage


def main():
    device = get_device()
    data_dir = f"{PROJECT_ROOT}/data/processed/beauty"

    # 1. 加载全量 embedding 和 cold item 列表
    emb = np.load(f"{data_dir}/item_emb.npy")        # (num_items+1, 384)
    with open(f"{data_dir}/cold_items.json") as f:
        cold_items = set(json.load(f))

    # 2. 只保留 warm item 的 embedding 参与训练
    #    item index 从1开始，构造 warm item 的 index 列表
    num_items = emb.shape[0] - 1
    warm_indices = [i for i in range(1, num_items + 1) if i not in cold_items]
    x = torch.tensor(emb[warm_indices], dtype=torch.float32)
    print(f"[data] warm items for RQ-VAE training: {x.shape[0]} "
          f"(excluded {len(cold_items)} cold items)")

    loader = DataLoader(TensorDataset(x), batch_size=256, shuffle=True)

    n_layers, num_codes = 3, 256
    model = RQVAE(in_dim=x.shape[1], latent_dim=32,
                  n_layers=n_layers, num_codes=num_codes, beta=0.25).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    n_epochs = 300
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_recon, total_vq = 0.0, 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            recon, indices, vq_loss = model(batch)
            recon_loss = mse(recon, batch)
            loss = recon_loss + vq_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_recon += recon_loss.item()
            total_vq += vq_loss.item()

        if epoch % 30 == 0:
            cos, usage = eval_metrics(model, x, device, n_layers, num_codes)
            usage_str = "/".join(str(u) for u in usage)
            print(f"[epoch {epoch}] recon={total_recon/len(loader):.5f} "
                  f"vq={total_vq/len(loader):.5f} | recon_cos={cos:.4f} "
                  f"| usage={usage_str} (of {num_codes})")

    os.makedirs(f"{PROJECT_ROOT}/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"{PROJECT_ROOT}/checkpoints/rqvae_warm.pt")
    print("[save] checkpoints/rqvae_warm.pt")


if __name__ == "__main__":
    main()