"""
S5c: GR 训练循环

标准 next-token prediction (和训语言模型一样):
  loss = CrossEntropy(logits, target)，ignore PAD 位置

中间监控:
  - token-level accuracy: 预测下一个 token 的准确率 (训练健康探针)
  - 分层准确率: 第1/2/3/4层分别的准确率 (预期第1层最准，体现层次)

产出: checkpoints/gr.pt  (S6 用它做 beam search 生成 + 正式评测)
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gr.dataset import GRDataset, train_collate, eval_collate, PAD, NUM_CODES, N_LAYERS
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
    """
    token 级准确率 (整体 + 分层)。
    分层: 每个 target token 属于第几层由它的 id 区间决定 (id//256)，
          但要排除 BOS/PAD。统计每层预测对的比例。
    """
    model.eval()
    correct = [0] * N_LAYERS
    total = [0] * N_LAYERS
    debug_printed = False
    for batch in loader:
        inp = batch["input"].to(device)
        tgt = batch["target"].to(device)
        logits = model(inp)                        # (B, L, V)
        pred = logits.argmax(dim=-1)               # (B, L)

        # 1. 只看非 PAD 的 target 位置
        valid = (tgt != PAD)

        # ===== 临时诊断:只打印第一个 batch =====
        if not debug_printed:
            # 1. 在同一批 logits 上手算 loss，和训练 loss 对比，确认 logits 没问题
            from torch.nn.functional import cross_entropy
            manual_loss = cross_entropy(logits.reshape(-1, logits.size(-1)),
                                        tgt.reshape(-1), ignore_index=PAD)
            print("  [debug] manual loss on this batch:", manual_loss.item())

            # 2. 找出第一个有效(非PAD)位置，打印它的 logits top-5
            B, L = tgt.shape
            flat_valid = valid.reshape(-1)
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_tgt = tgt.reshape(-1)
            first_valid = flat_valid.nonzero()[0].item()
            top5 = flat_logits[first_valid].topk(5)
            # print("  [debug] first valid pos target:", flat_tgt[first_valid].item())
            # print("  [debug] its top5 pred tokens:", top5.indices.tolist())
            # print("  [debug] its top5 logits:", [round(v, 2) for v in top5.values.tolist()])
            debug_printed = True
        # ========================================

        # 2. 每个 target token 的层号 = id // 256 (0~3)，BOS(1024)/PAD(1025) 会 >=4，被 valid 排除
        layer = tgt // NUM_CODES                   # (B, L)
        hit = (pred == tgt) & valid

        for l in range(N_LAYERS):
            layer_mask = valid & (layer == l)
            total[l] += layer_mask.sum().item()
            correct[l] += (hit & (layer == l)).sum().item()

    overall = sum(correct) / max(sum(total), 1)
    per_layer = [correct[l] / max(total[l], 1) for l in range(N_LAYERS)]
    return overall, per_layer


def main():
    device = get_device()

    train_ds = GRDataset(split="train", max_items=50)
    val_ds = GRDataset(split="val", max_items=50)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                              collate_fn=train_collate)
    # 用 train_collate 复用:token_accuracy 需要 input/target，val 也临时用 train 格式
    val_for_acc = GRDataset(split="train", max_items=50)  # 见下方说明
    val_acc_loader = DataLoader(val_ds, batch_size=128, collate_fn=eval_collate)

    model = GRModel(max_len=train_ds.max_tokens, d_model=128,
                    n_heads=4, n_blocks=4, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    # 1. CrossEntropy 自动忽略 PAD 位置，不用手动 mask
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    n_epochs = 100
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            inp = batch["input"].to(device)
            tgt = batch["target"].to(device)
            logits = model(inp)                    # (B, L, V)
            # 2. reshape 成 (B*L, V) 和 (B*L,) 算交叉熵
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            # 加这行:一旦loss变NaN立刻报警,别等训练跑完
            assert not torch.isnan(loss), "loss is NaN! check padding mask"

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 防梯度爆炸
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            # 3. 用训练集算 token 准确率 (健康探针)
            overall, per_layer = token_accuracy(model, train_loader, device)
            pl = "/".join(f"{a:.3f}" for a in per_layer)
            print(f"[epoch {epoch}] loss={total_loss/len(train_loader):.4f} "
                  f"| token_acc={overall:.3f} | per-layer(1/2/3/4)={pl}")

    os.makedirs(f"{PROJECT_ROOT}/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"{PROJECT_ROOT}/checkpoints/gr.pt")
    print("[save] checkpoints/gr.pt")


if __name__ == "__main__":
    main()