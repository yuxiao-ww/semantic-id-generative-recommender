"""
S7b-3: 在 warm 序列上重训 GR (冷启动实验用)

和 S5 的 gr/train.py 唯一区别:
  - 序列用 sequences_warm.json (不含cold item)
  - SID 用 item_sid_cold.json (含cold item的SID，来自warm-RQVAE)

训练序列全是warm item，但SID词表覆盖全部item(含cold)，
这样推理时GR才能生成cold item的SID token。

产出: checkpoints/gr_warm.pt
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gr.dataset import GRDataset, train_collate, PAD, NUM_CODES, N_LAYERS
from gr.model import GRModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def token_accuracy(model, loader, device):
    """复用S5的分层token准确率监控。"""
    model.eval()
    correct = [0] * N_LAYERS
    total = [0] * N_LAYERS
    for batch in loader:
        inp = batch["input"].to(device)
        tgt = batch["target"].to(device)
        logits = model(inp)
        pred = logits.argmax(dim=-1)
        valid = (tgt != PAD)
        layer = tgt // NUM_CODES
        hit = (pred == tgt) & valid
        for l in range(N_LAYERS):
            lm = valid & (layer == l)
            total[l] += lm.sum().item()
            correct[l] += (hit & (layer == l)).sum().item()
    overall = sum(correct) / max(sum(total), 1)
    per_layer = [correct[l] / max(total[l], 1) for l in range(N_LAYERS)]
    return overall, per_layer


def main():
    device = get_device()

    # 1. 关键: 用 warm序列 + cold SID (含cold item的SID)
    train_ds = GRDataset(split="train", max_items=50,
                         seq_file="sequences_warm.json",
                         sid_file="item_sid_cold.json")
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                              collate_fn=train_collate)
    print(f"[data] warm training sequences: {len(train_ds)}")

    model = GRModel(max_len=train_ds.max_tokens, d_model=128,
                    n_heads=4, n_blocks=4, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    n_epochs = 100
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            inp = batch["input"].to(device)
            tgt = batch["target"].to(device)
            logits = model(inp)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            assert not torch.isnan(loss), "loss is NaN!"
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if epoch % 20 == 0:
            overall, per_layer = token_accuracy(model, train_loader, device)
            pl = "/".join(f"{a:.3f}" for a in per_layer)
            print(f"[epoch {epoch}] loss={total_loss/len(train_loader):.4f} "
                  f"| token_acc={overall:.3f} | per-layer={pl}")

    os.makedirs(f"{PROJECT_ROOT}/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"{PROJECT_ROOT}/checkpoints/gr_warm.pt")
    print("[save] checkpoints/gr_warm.pt")


if __name__ == "__main__":
    main()