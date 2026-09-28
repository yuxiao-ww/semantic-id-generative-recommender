"""
S6-2: GR 全量评测

对每个test用户 beam search 生成有序推荐列表，
answer在列表中的位置 = rank，套用和S2b一致的Recall@K/NDCG@K公式。

和SASRec指标定义完全一致，结果可直接对表。
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from gr.dataset import GRDataset, eval_collate
from gr.model import GRModel
from gr.generate import build_sid_to_item, generate_recommendations

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def evaluate_gr(model, loader, sid2item, device, beam_width=50, ks=(5, 10, 20)):
    """
    beam_width: 生成的候选数，要 >> max(ks) 以留invalid过滤余量
    """
    model.eval()
    max_k = max(ks)
    recalls = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}
    total_invalid, total_gen = 0, 0

    for batch in loader:
        inp = batch["input"].to(device)
        answers = batch["answer_item"]                # list[int]

        # 1. 对整个batch生成推荐列表 (每个用户一个有序item列表)
        rec_lists, invalid, total = generate_recommendations(
            model, inp, beam_width, sid2item, device)
        total_invalid += invalid
        total_gen += total

        # 2. 每个用户: 找answer在推荐列表里的位置 = rank
        for b in range(len(rec_lists)):
            answer = answers[b]
            recs = rec_lists[b]
            # 3. answer在列表第几位 (0-indexed -> rank从1开始)
            if answer in recs:
                rank = recs.index(answer) + 1         # 1-indexed
            else:
                rank = max_k + 1                      # 没生成出来，视为排在K之外
            # 4. 套用和S2b一致的公式
            for k in ks:
                hit = 1.0 if rank <= k else 0.0
                ndcg = (1.0 / np.log2(rank + 1)) if rank <= k else 0.0
                recalls[k].append(hit)
                ndcgs[k].append(ndcg)

    result = {}
    for k in ks:
        result[f"Recall@{k}"] = float(np.mean(recalls[k]))
        result[f"NDCG@{k}"] = float(np.mean(ndcgs[k]))
    invalid_rate = total_invalid / max(total_gen, 1)
    return result, invalid_rate


def main():
    device = get_device()
    data_dir = f"{PROJECT_ROOT}/data/processed/beauty"

    test_ds = GRDataset(split="test", max_items=50)

    # 1. 随机采样 3000 用户评测 (全量太慢，采样指标已足够稳定)
    import random
    random.seed(42)  # 固定种子，可复现
    if len(test_ds.seqs) > 3000:
        test_ds.seqs = random.sample(test_ds.seqs, 3000)

    test_loader = DataLoader(test_ds, batch_size=128, collate_fn=eval_collate)

    model = GRModel(max_len=test_ds.max_tokens, d_model=128,
                    n_heads=4, n_blocks=4).to(device)
    model.load_state_dict(torch.load(f"{PROJECT_ROOT}/checkpoints/gr.pt",
                                     map_location=device))
    model.eval()

    sid2item = build_sid_to_item(f"{data_dir}/item_sid.json")

    # beam=50: 要算到Recall@20，加上17% invalid，50足够留余量
    # result, invalid_rate = evaluate_gr(model, test_loader, sid2item, device,
    #                                    beam_width=50, ks=(5, 10, 20))
    result, invalid_rate = evaluate_gr(model, test_loader, sid2item, device,
                                        beam_width=20, ks=(5, 10, 20))

    print("=" * 55)
    print("GR (Generative Recommender) — test set")
    for k in (5, 10, 20):
        print(f"  Recall@{k}: {result[f'Recall@{k}']:.4f} | "
              f"NDCG@{k}: {result[f'NDCG@{k}']:.4f}")
    print(f"  invalid generation rate: {invalid_rate*100:.1f}%")
    print("=" * 55)
    print("SASRec baseline (from S2):")
    print("  Recall@10: 0.0557 | NDCG@10: 0.0298")
    print("=" * 55)


if __name__ == "__main__":
    main()