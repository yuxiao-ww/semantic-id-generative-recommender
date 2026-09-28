"""
S7b-4: 在 warm 序列上重训 SASRec (冷启动对比basis)

关键设计 (方案a):
  - embedding表覆盖全部item(num_items=12101,含cold)
  - 但训练序列不含cold item -> cold item的embedding保持随机初始化
  - 模拟"新item进系统但无交互数据"的真实冷启动状态

产出: checkpoints/sasrec_warm.pt
"""

import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import SeqRecDataset, eval_collate
from baselines.sasrec import SASRec
from baselines.train_sasrec import train_collate, compute_loss

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    device = get_device()
    data_dir = f"{PROJECT_ROOT}/data/processed/beauty"

    # 1. 真实item总数(含cold): 从全量embedding推断，不是从warm序列
    emb_path = f"{data_dir}/item_emb.npy"
    import numpy as np
    full_num_items = np.load(emb_path).shape[0] - 1     # 12101
    print(f"[data] full num_items (incl. cold): {full_num_items}")

    # 2. warm序列训练，但强制num_items为全量(cold item有随机embedding槽位)
    train_ds = SeqRecDataset(data_dir, split="train", max_len=50,
                             seq_file="sequences_warm.json")
    train_ds.set_num_items(full_num_items)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                              collate_fn=train_collate)
    print(f"[data] warm sequences: {len(train_ds)}")

    model = SASRec(num_items=full_num_items, max_len=50, d_model=64,
                   n_heads=1, n_blocks=2, dropout=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98))

    n_epochs = 100
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            inp = batch["input"].to(device)
            tgt = batch["target"].to(device)
            loss = compute_loss(model, inp, tgt, full_num_items)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 20 == 0:
            print(f"[epoch {epoch}] loss={total_loss/len(train_loader):.4f}")

    os.makedirs(f"{PROJECT_ROOT}/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"{PROJECT_ROOT}/checkpoints/sasrec_warm.pt")
    print("[save] checkpoints/sasrec_warm.pt")


if __name__ == "__main__":
    main()