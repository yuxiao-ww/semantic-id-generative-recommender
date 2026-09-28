"""
S7b-2: 用 warm-RQVAE 给全部item(含cold)生成SID

核心: rqvae_warm 从没见过cold item，但因为它学的是通用"语义->SID"映射，
能给cold item生成语义合理的SID。这是冷启动可行的技术基础。

产出:
  item_sid_cold.json  全部item的4位SID (warm用见过的编码，cold用泛化编码)
验证:
  cold item的SID第一层token，是否和语义相近的warm item一致
"""

import json
import os
from collections import defaultdict

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

    emb = np.load(f"{data_dir}/item_emb.npy")
    num_items = emb.shape[0] - 1
    with open(f"{data_dir}/cold_items.json") as f:
        cold_items = set(json.load(f))
    with open(f"{data_dir}/item_meta.json") as f:
        meta = json.load(f)

    # 1. 加载只见过warm的RQVAE
    model = RQVAE(in_dim=emb.shape[1], latent_dim=32,
                  n_layers=3, num_codes=256, beta=0.25).to(device)
    model.load_state_dict(torch.load(f"{PROJECT_ROOT}/checkpoints/rqvae_warm.pt",
                                     map_location=device))
    model.eval()

    # 2. 编码全部item(warm+cold都用同一个warm-RQVAE)
    x = torch.tensor(emb[1:], dtype=torch.float32).to(device)
    sids = model.get_semantic_ids(x).cpu().numpy()      # (num_items, 3)

    # 3. 生成3位SID，再加去重位(逻辑同S4b)
    raw = {str(i + 1): sids[i].tolist() for i in range(num_items)}
    sid_to_items = defaultdict(list)
    for item_idx in sorted(raw.keys(), key=int):
        sid_to_items[tuple(raw[item_idx])].append(item_idx)
    item_sid = {}
    for sid3, items in sid_to_items.items():
        for dedup, item_idx in enumerate(items):
            item_sid[item_idx] = list(sid3) + [dedup]

    with open(f"{data_dir}/item_sid_cold.json", "w") as f:
        json.dump(item_sid, f)
    print(f"[save] item_sid_cold.json ({len(item_sid)} items)")

    # 4. 验证冷启动泛化: cold item的SID第一层token，是否落在
    #    "语义相近的warm item"同一个k1桶里
    #    做法: 对每个k1桶，统计里面warm和cold item的主品类是否一致
    def main_cat(i):
        cats = meta[str(i)]["categories"]
        return cats[1] if len(cats) >= 2 else (cats[0] if cats else "Unknown")

    # 4.1 按k1分桶，看cold item是否和同桶warm item品类一致
    k1_buckets = defaultdict(lambda: {"warm": [], "cold": []})
    for item_idx, sid in item_sid.items():
        k1 = sid[0]
        key = "cold" if int(item_idx) in cold_items else "warm"
        k1_buckets[k1][key].append(int(item_idx))

    # 4.2 对有cold item的桶，检查cold item品类是否匹配桶内warm主品类
    match, total = 0, 0
    for k1, groups in k1_buckets.items():
        if not groups["cold"] or not groups["warm"]:
            continue
        # 桶内warm item的主品类(众数)
        from collections import Counter
        warm_cats = Counter(main_cat(i) for i in groups["warm"])
        dominant_cat = warm_cats.most_common(1)[0][0]
        # 每个cold item的品类是否等于桶主品类
        for ci in groups["cold"]:
            total += 1
            if main_cat(ci) == dominant_cat:
                match += 1

    print(f"[cold-SID validation] cold items whose category matches "
          f"their k1-bucket's dominant warm category: "
          f"{match}/{total} = {100*match/max(total,1):.1f}%")

    # 5. 打印几个cold item和它们的k1桶邻居做人肉检查
    print("\n[sample] cold items and their k1-bucket warm neighbors:")
    shown = 0
    for k1, groups in k1_buckets.items():
        if groups["cold"] and groups["warm"] and shown < 3:
            ci = groups["cold"][0]
            wi = groups["warm"][0]
            print(f"  k1={k1}:")
            print(f"    COLD: {meta[str(ci)]['title'][:55]}")
            print(f"    WARM neighbor: {meta[str(wi)]['title'][:55]}")
            shown += 1


if __name__ == "__main__":
    main()