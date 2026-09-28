"""
S4b: 冲突消解 —— 给每个 SID 追加第 4 位去重码

读:  item_sid_raw.json  (3位SID，可能冲突)
写:  item_sid.json       (4位SID，全局唯一) —— GR 的最终输入

第4位职责: 只区分语义完全相同 (前3位撞车) 的 item，不承载语义。
"""

import json
import os
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    data_dir = f"{PROJECT_ROOT}/data/processed/beauty"
    with open(f"{data_dir}/item_sid_raw.json") as f:
        raw = json.load(f)

    # 1. 按 (k1,k2,k3) 分桶。注意要按 item_idx 排序遍历，
    #    保证同一个 item 每次跑都拿到相同的第4位 (可复现)
    sid_to_items = defaultdict(list)
    for item_idx in sorted(raw.keys(), key=int):
        sid_to_items[tuple(raw[item_idx])].append(item_idx)

    # 2. 桶内顺序编号，追加第4位
    item_sid = {}
    max_dedup = 0
    for sid3, items in sid_to_items.items():
        for dedup_code, item_idx in enumerate(items):
            item_sid[item_idx] = list(sid3) + [dedup_code]
            max_dedup = max(max_dedup, dedup_code)

    # 3. 验证全局唯一 (消解后不该再有任何冲突)
    all_sids = [tuple(v) for v in item_sid.values()]
    assert len(all_sids) == len(set(all_sids)), "still has collision!"

    print(f"[resolve] total items: {len(item_sid)}")
    print(f"[resolve] max dedup code used: {max_dedup} "
          f"(so 4th position ranges 0..{max_dedup})")
    print(f"[resolve] all SIDs unique: True")

    with open(f"{data_dir}/item_sid.json", "w") as f:
        json.dump(item_sid, f)
    print(f"[save] item_sid.json (final, collision-free)")


if __name__ == "__main__":
    main()