"""
S4c: 语义可视化 —— 验证 SID 第一层 token (k1) 对应可解释的品类

两个证据:
  1. 抽样打印: 几个 k1 桶里的 item title，人肉看语义一致性
  2. 品类纯度: 每个 k1 桶里 item 的主品类占比，量化语义有效性

产出面试/简历用的"Semantic ID 有语义层次"的硬证据。
"""

import json
import os
from collections import Counter, defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_main_category(cats):
    """
    从 category 路径取一个有代表性的品类做纯度统计。
    Beauty 数据的 categories 形如 ["Beauty","Makeup","Lips","Lipstick"]，
    第 0 个通常是 "Beauty" (所有 item 都一样，没区分度)，
    所以取第 1 个 (二级品类) 作为主品类 —— 它才有区分度。
    """
    if not cats or len(cats) < 2:
        return cats[0] if cats else "Unknown"
    return cats[1]


def main():
    data_dir = f"{PROJECT_ROOT}/data/processed/beauty"
    with open(f"{data_dir}/item_sid.json") as f:
        item_sid = json.load(f)
    with open(f"{data_dir}/item_meta.json") as f:
        meta = json.load(f)

    # 1. 按 k1 (第一层 token) 分组
    k1_buckets = defaultdict(list)
    for item_idx, sid in item_sid.items():
        k1 = sid[0]
        k1_buckets[k1].append(item_idx)

    print(f"[viz] number of non-empty k1 buckets: {len(k1_buckets)}")

    # 2. 抽样打印: 挑几个 item 数适中的桶 (太小没代表性，太大打印刷屏)
    #    按桶大小排序，取中间几个
    sorted_buckets = sorted(k1_buckets.items(), key=lambda kv: len(kv[1]))
    mid = len(sorted_buckets) // 2
    sample_buckets = sorted_buckets[mid:mid + 3]     # 中位数附近3个桶

    for k1, items in sample_buckets:
        print(f"\n=== k1={k1}  ({len(items)} items) ===")
        # 打印前 8 个 item 的 title
        for item_idx in items[:8]:
            title = meta[item_idx]["title"][:55]
            cat = get_main_category(meta[item_idx]["categories"])
            print(f"  [{cat}] {title}")

    # 3. 品类纯度: 每个桶里主品类占比，再对所有桶取加权平均
    #    纯度高 = k1 确实对应某个品类，语义有效
    total_items = 0
    weighted_purity = 0.0
    per_bucket_purity = []
    for k1, items in k1_buckets.items():
        if len(items) < 5:                           # 太小的桶纯度不稳定，跳过统计
            continue
        cats = [get_main_category(meta[i]["categories"]) for i in items]
        most_common_cat, count = Counter(cats).most_common(1)[0]
        purity = count / len(items)
        per_bucket_purity.append((k1, len(items), most_common_cat, purity))
        weighted_purity += count
        total_items += len(items)

    avg_purity = weighted_purity / total_items
    print(f"\n[viz] weighted avg category purity (k1 buckets >=5 items): "
          f"{avg_purity:.3f}")

    # 4. 打印几个高纯度桶做展示 (纯度排序取 top)
    per_bucket_purity.sort(key=lambda x: -x[3])
    print(f"[viz] top-5 purest k1 buckets:")
    for k1, size, cat, purity in per_bucket_purity[:5]:
        print(f"       k1={k1:3d} | {size:4d} items | {purity:.2f} are '{cat}'")


if __name__ == "__main__":
    main()