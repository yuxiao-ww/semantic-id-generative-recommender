"""
S2b: 全量排序评测器 (Recall@K / NDCG@K)

被 SASRec / GR / 所有消融复用，是全项目最核心的基础设施。

核心逻辑只有一件事: 算出正确答案 (answer) 在全量排序里的 rank，
Recall 和 NDCG 都由 rank 直接得出。

三个必须做对的点:
  1. 全量排序: 对所有 item 打分，不采样负样本 (采样指标不可比)
  2. mask PAD(0) 和 seen item: 否则 rank 虚高
  3. rank 用"有多少 item 分数 >= answer 分数"来算，处理并列分数要小心
"""

import numpy as np
import torch


@torch.no_grad()
def evaluate(model, loader, num_items, device, ks=(5, 10, 20)):
    """
    model: 需实现 score_all(input_batch) -> (B, num_items+1) 的打分接口
           约定所有序列推荐模型都暴露这个方法，评测器就能通用
    loader: val/test 的 DataLoader (用 eval_collate)
    ks: 要算的 top-K 列表
    """
    model.eval()
    max_k = max(ks)

    # 1. 累加器: 每个 K 存一个 recall 列表、一个 ndcg 列表，最后取均值
    recalls = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}

    for batch in loader:
        inp = batch["input"].to(device)          # (B, L)
        answers = batch["answer"].to(device)      # (B,)
        seen_list = batch["seen"]                 # list[set]，长度 B

        # 2. 全量打分: 模型对所有 item 输出分数
        scores = model.score_all(inp)             # (B, num_items+1)

        # 3. mask PAD: index 0 永远不该被推荐，分数打到负无穷
        scores[:, 0] = float("-inf")

        # 4. mask 已交互 item: 每个用户把自己 seen 过的 item 分数打到负无穷
        #    注意 answer 本身不在 input 里 (leave-one-out 切掉了)，但可能在
        #    seen 里 (seen 是全序列)，所以要先把 answer 从 mask 中排除，
        #    否则会把正确答案也 mask 掉，rank 永远算不对
        for b in range(inp.size(0)):
            ans = answers[b].item()
            for it in seen_list[b]:
                if it != ans:                     # 保护正确答案
                    scores[b, it] = float("-inf")

        # 5. 算 rank: answer 的分数
        #    rank = 1 + (有多少 item 分数严格大于 answer 分数)
        #    用严格大于 (>) 而非 >=，避免并列时把自己也算进去
        ans_scores = scores[torch.arange(inp.size(0)), answers]   # (B,)
        # (B, num_items+1) 每行有多少个分数 > answer 的分数
        rank = (scores > ans_scores.unsqueeze(1)).sum(dim=1) + 1  # (B,) 1-indexed

        rank = rank.cpu().numpy()

        # 6. 由 rank 直接算指标
        for k in ks:
            hit = (rank <= k).astype(np.float32)          # Recall@K: 命中为 1
            # NDCG@K: 命中则 1/log2(rank+1)，否则 0
            ndcg = np.where(rank <= k, 1.0 / np.log2(rank + 1), 0.0)
            recalls[k].extend(hit.tolist())
            ndcgs[k].extend(ndcg.tolist())

    # 7. 对所有用户取平均
    result = {}
    for k in ks:
        result[f"Recall@{k}"] = float(np.mean(recalls[k]))
        result[f"NDCG@{k}"] = float(np.mean(ndcgs[k]))
    return result


def print_metrics(result, tag=""):
    """整齐打印，方便和论文对表。"""
    line = " | ".join(f"{k}: {v:.4f}" for k, v in result.items())
    print(f"[{tag}] {line}" if tag else f"[eval] {line}")
