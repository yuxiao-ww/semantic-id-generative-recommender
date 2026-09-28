"""
S3a 训练: 用重建损失训练纯 AE。

loss = MSE(recon, original) —— 让重建向量逼近原始语义向量。
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from semantic_id.rqvae import AutoEncoder

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def recon_cosine(model, x, device):
    """
    验收指标: 重建向量和原向量的平均余弦相似度。
    比 MSE 更直观 —— 1.0 = 完美重建，0 = 毫不相关。
    """
    model.eval()
    recon, _ = model(x.to(device))
    recon = torch.nn.functional.normalize(recon, dim=1)
    orig = torch.nn.functional.normalize(x.to(device), dim=1)
    return (recon * orig).sum(dim=1).mean().item()


def main():
    device = get_device()
    data_dir = f"{PROJECT_ROOT}/data/processed/beauty"

    # 1. 加载 S1 的 embedding，去掉第 0 行 (PAD 占位，全 0，不参与训练)
    emb = np.load(f"{data_dir}/item_emb.npy")        # (num_items+1, 384)
    emb = emb[1:]                                     # (num_items, 384)
    x = torch.tensor(emb, dtype=torch.float32)
    print(f"[data] training on {x.shape[0]} item embeddings, dim={x.shape[1]}")

    dataset = TensorDataset(x)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = AutoEncoder(in_dim=x.shape[1], latent_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    n_epochs = 200
    for epoch in range(1, n_epochs + 1):
        model.train()
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            recon, z = model(batch)
            loss = mse(recon, batch)                 # 重建损失
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        if epoch % 20 == 0:
            cos = recon_cosine(model, x, device)
            print(f"[epoch {epoch}] mse={total/len(loader):.5f} | recon_cos={cos:.4f}")

    # 2. 存权重 (S3b 会参考这个结构，也方便对比量化前后的重建质量)
    os.makedirs(f"{PROJECT_ROOT}/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"{PROJECT_ROOT}/checkpoints/ae.pt")
    print("[save] checkpoints/ae.pt")


if __name__ == "__main__":
    main()