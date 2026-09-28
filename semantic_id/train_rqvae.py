"""
S3b-2: 端到端训练 RQVAE

总 loss = 重建loss + 量化loss
核心监控: codebook 使用率 (每层用到多少个码字) —— 塌缩的照妖镜

产出: checkpoints/rqvae.pt (S4 用它给所有 item 生成 Semantic ID)
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from semantic_id.rqvae import RQVAE

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def eval_metrics(model, x, device, n_layers, num_codes):
    """
    两个验收指标:
      1. recon_cos: 重建质量 (对比 S3a 的 0.9246 上界)
      2. usage: 每层码本的使用率 (塌缩检测)
    """
    model.eval()
    recon, indices, _ = model(x.to(device))    # indices: (N, n_layers)

    # 1. 重建余弦相似度
    recon_n = torch.nn.functional.normalize(recon, dim=1)
    orig_n = torch.nn.functional.normalize(x.to(device), dim=1)
    cos = (recon_n * orig_n).sum(1).mean().item()

    # 2. 每层唯一码字数: 统计每一列 (每层) 用到了几个不同的码字
    usage = []
    for layer in range(n_layers):
        n_unique = indices[:, layer].unique().numel()
        usage.append(n_unique)
    return cos, usage


def main():
    device = get_device()
    data_dir = f"{PROJECT_ROOT}/data/processed/beauty"

    emb = np.load(f"{data_dir}/item_emb.npy")[1:]   # 去掉PAD行
    x = torch.tensor(emb, dtype=torch.float32)
    print(f"[data] {x.shape[0]} items, dim={x.shape[1]}")

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
            loss = recon_loss + vq_loss        # 总loss = 重建 + 量化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_recon += recon_loss.item()
            total_vq += vq_loss.item()

        if epoch % 20 == 0:
            cos, usage = eval_metrics(model, x, device, n_layers, num_codes)
            usage_str = "/".join(str(u) for u in usage)
            print(f"[epoch {epoch}] recon={total_recon/len(loader):.5f} "
                  f"vq={total_vq/len(loader):.5f} | recon_cos={cos:.4f} "
                  f"| codebook usage(per layer)={usage_str} (of {num_codes})")

    # 保存: S4 生成 Semantic ID 要用
    os.makedirs(f"{PROJECT_ROOT}/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"{PROJECT_ROOT}/checkpoints/rqvae.pt")
    print("[save] checkpoints/rqvae.pt")


if __name__ == "__main__":
    main()