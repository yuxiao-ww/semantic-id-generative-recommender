"""
S2c-3: SASRec 训练循环

训练目标: next-item prediction + 负采样 + BCE loss (原论文用法)
  每个位置的正样本 = target[t]，随机采一个负样本
  BCE 推高正样本分数、压低负样本分数

产出: 训练好的 SASRec，在 test 上评测得到主 baseline 数字
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import SeqRecDataset, eval_collate
from baselines.sasrec import SASRec
from eval.evaluator import evaluate, print_metrics

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_collate(batch):
    """train split 的 collate: input / target 都是定长张量，直接 stack。"""
    return {
        "input": torch.stack([b["input"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
    }


def compute_loss(model, input_ids, target_ids, num_items):
    """
    负采样 + BCE loss。
    input_ids/target_ids: (B, L)，target 是 input 错位一位的下一个 item
    """
    B, L = input_ids.shape
    device = input_ids.device

    # 1. forward 拿每个位置的 hidden state
    h = model(input_ids)                             # (B, L, d)

    # 2. 正样本 embedding: target 位置的 item embedding
    pos_emb = model.item_emb(target_ids)             # (B, L, d)

    # 3. 负采样: 每个位置随机采一个 item id (1..num_items)
    #    简化版随机负采样，不排除用户历史 (稀疏数据下撞上的概率极低，
    #    SASRec 原实现也是这么做的)
    neg_ids = torch.randint(1, num_items + 1, (B, L), device=device)
    neg_emb = model.item_emb(neg_ids)                # (B, L, d)

    # 4. 打分: hidden state 和 pos/neg embedding 逐位置内积
    pos_score = (h * pos_emb).sum(dim=-1)            # (B, L)
    neg_score = (h * neg_emb).sum(dim=-1)            # (B, L)

    # 5. mask: 只在 target 非 PAD 的位置算 loss
    #    (左 padding 的位置 target=0，是补出来的，不能算进 loss)
    mask = (target_ids != 0).float()                 # (B, L)

    # 6. BCE loss: -log σ(pos) - log(1-σ(neg))
    #    用 logsigmoid 数值稳定，避免手动 sigmoid 后取 log 溢出
    pos_loss = -torch.nn.functional.logsigmoid(pos_score) * mask
    neg_loss = -torch.nn.functional.logsigmoid(-neg_score) * mask
    # 7. 对有效位置求和再除以有效位置数 -> 平均 loss
    loss = (pos_loss + neg_loss).sum() / mask.sum()
    return loss


def main():
    device = get_device()
    data_dir = f"{PROJECT_ROOT}/data/processed/beauty"

    train_ds = SeqRecDataset(data_dir, split="train", max_len=50)
    val_ds = SeqRecDataset(data_dir, split="val", max_len=50)
    test_ds = SeqRecDataset(data_dir, split="test", max_len=50)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                              collate_fn=train_collate)
    val_loader = DataLoader(val_ds, batch_size=256, collate_fn=eval_collate)
    test_loader = DataLoader(test_ds, batch_size=256, collate_fn=eval_collate)

    num_items = train_ds.num_items
    model = SASRec(num_items=num_items, max_len=50, d_model=64,
                   n_heads=1, n_blocks=2, dropout=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98))

    n_epochs = 100
    best_val = -1.0
    best_state = None
    patience, bad = 20, 0    # early stopping: 20 个 epoch 不涨就停

    for epoch in range(1, n_epochs + 1):
        # 1. 训练一个 epoch
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            inp = batch["input"].to(device)
            tgt = batch["target"].to(device)
            loss = compute_loss(model, inp, tgt, num_items)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # 2. 每 5 个 epoch 在 val 上评一次，用 NDCG@10 做早停指标
        if epoch % 5 == 0:
            val_result = evaluate(model, val_loader, num_items, device, ks=(10,))
            val_ndcg = val_result["NDCG@10"]
            print(f"[epoch {epoch}] loss={avg_loss:.4f} | val NDCG@10={val_ndcg:.4f}")

            # 3. 保存最好的模型 (以 val 为准，绝不看 test)
            if val_ndcg > best_val:
                best_val = val_ndcg
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience // 5:
                    print(f"[early stop] no improvement for {patience} epochs")
                    break

    # 4. 加载最优权重，在 test 上出最终数字
    model.load_state_dict(best_state)
    test_result = evaluate(model, test_loader, num_items, device, ks=(5, 10, 20))
    print_metrics(test_result, tag="SASRec (test)")

    # 5. 存模型，后面 GR 对比要引用这个数
    os.makedirs(f"{PROJECT_ROOT}/checkpoints", exist_ok=True)
    torch.save(best_state, f"{PROJECT_ROOT}/checkpoints/sasrec.pt")
    print(f"[save] checkpoints/sasrec.pt")


if __name__ == "__main__":
    main()
