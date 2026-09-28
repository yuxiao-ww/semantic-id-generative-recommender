"""
S4a: 用训好的 RQVAE 给全量 item 生成 Semantic ID

产出:
  data/processed/beauty/item_sid.json   {item_idx: [k1, k2, k3]}
统计:
  冲突率 —— 多少 item 共享同一个 (k1,k2,k3)，决定 S4b 怎么消解

这一步只生成 + 统计，不做冲突消解 (先测量严重程度再决定方案)。
"""

import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch

from semantic_id.rqvae import RQVAE

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def main():
    device = get_device()
    data_dir = f"{PROJECT_ROOT}/data/processed/beauty"

    # 1. 加载 S1 embedding，第 0 行是 PAD 占位，item 从 index 1 开始
    emb = np.load(f"{data_dir}/item_emb.npy")        # (num_items+1, 384)
    num_items = emb.shape[0] - 1

    # 2. 重建 RQVAE 结构并加载权重 (超参必须和 train_rqvae.py 完全一致)
    model = RQVAE(in_dim=emb.shape[1], latent_dim=32,
                  n_layers=3, num_codes=256, beta=0.25).to(device)
    model.load_state_dict(torch.load(f"{PROJECT_ROOT}/checkpoints/rqvae.pt",
                                     map_location=device))
    model.eval()

    # 3. 给 item 1..num_items 生成 SID (跳过 index 0 的 PAD)
    x = torch.tensor(emb[1:], dtype=torch.float32).to(device)
    sids = model.get_semantic_ids(x)                 # (num_items, 3)
    sids = sids.cpu().numpy()

    # 4. 存成 {item_idx: [k1,k2,k3]}，item_idx 从 1 开始对齐
    item_sid = {str(i + 1): sids[i].tolist() for i in range(num_items)}

    # 5. 冲突统计: 把每个 SID 元组当 key，看多少 item 落在同一个 key
    sid_to_items = defaultdict(list)
    for item_idx, sid in item_sid.items():
        sid_to_items[tuple(sid)].append(item_idx)

    total_items = len(item_sid)
    unique_sids = len(sid_to_items)
    # 落在"有冲突的桶"里的 item 数 (桶里 item 数 > 1)
    collided_items = sum(len(v) for v in sid_to_items.values() if len(v) > 1)
    # 桶大小分布: 看最挤的桶有多少 item
    bucket_sizes = Counter(len(v) for v in sid_to_items.values())
    max_bucket = max(len(v) for v in sid_to_items.values())

    print(f"[SID] total items: {total_items}")
    print(f"[SID] unique (k1,k2,k3): {unique_sids}")
    print(f"[SID] items in collision (bucket>1): {collided_items} "
          f"({100*collided_items/total_items:.1f}%)")
    print(f"[SID] largest bucket: {max_bucket} items share one SID")
    print(f"[SID] bucket size distribution (size: count):")
    for size in sorted(bucket_sizes.keys()):
        print(f"       {size} items -> {bucket_sizes[size]} buckets")

    # 6. 存 SID (未消解版本，S4b 会在此基础上加去重位)
    with open(f"{data_dir}/item_sid_raw.json", "w") as f:
        json.dump(item_sid, f)
    print(f"[save] item_sid_raw.json (before collision resolution)")


if __name__ == "__main__":
    main()