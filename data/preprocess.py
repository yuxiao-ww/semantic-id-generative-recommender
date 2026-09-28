"""
Stage 0: Amazon Reviews 2014 -> 用户交互序列

产出:
  data/processed/beauty/sequences.json
  data/processed/beauty/item_meta.json
  data/processed/beauty/stats.json
"""

import ast
import gzip
import json
import os
from collections import defaultdict

from pathlib import Path

# 以本文件位置为锚点推出项目根目录，不依赖 CWD / IDE 配置。
# preprocess.py 在 <root>/data/ 下，所以 parent.parent 才是根目录。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed"


def parse_gz(path):
    """
    逐行读 .gz。
    坑点: reviews_*.json.gz 是合法 JSON，但 meta_*.json.gz 是 Python dict
    字面量（单引号），json.loads 会直接抛异常。所以先试 json，失败退化到
    ast.literal_eval。这是 Amazon 2014 数据集最经典的一个坑。
    """
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield ast.literal_eval(line)


def load_interactions(review_path):
    """
    只取三个字段: user / item / timestamp。
    故意不用 overall(评分): 序列推荐的标准协议是把"有交互"当隐式正反馈，
    不做 rating>=4 之类的过滤，否则结果和论文不可比。
    """
    inter = []
    for r in parse_gz(review_path):
        inter.append((r["reviewerID"], r["asin"], int(r["unixReviewTime"])))
    return inter


def dedup(inter):
    """同一 (user, item) 可能有多条评论，只保留最早的一次。"""
    best = {}
    for u, i, t in inter:
        if (u, i) not in best or t < best[(u, i)]:
            best[(u, i)] = t
    return [(u, i, t) for (u, i), t in best.items()]


def k_core(inter, k=5, max_iter=20):
    """
    迭代 k-core 过滤。
    为什么必须迭代: 删掉低频 item 后，某些 user 的交互数会掉到 k 以下，
    反过来也一样，必须来回删到不动点。
    官方 *_5.json.gz 本身已是 5-core，正常情况第 0 轮就收敛；
    但上面去重可能破坏它，所以这一步保留作为保险。
    """
    for it in range(max_iter):
        ucnt, icnt = defaultdict(int), defaultdict(int)
        for u, i, _ in inter:
            ucnt[u] += 1
            icnt[i] += 1
        kept = [(u, i, t) for u, i, t in inter if ucnt[u] >= k and icnt[i] >= k]
        if len(kept) == len(inter):
            print(f"[k-core] converged at iteration {it}")
            return kept
        inter = kept
    print("[k-core] WARNING: hit max_iter without convergence")
    return inter


def build_sequences(inter):
    """
    按 user 分组，时间升序。
    tie-break 用 asin 字符串: 时间戳相同的交互如果不定序，换台机器
    读入顺序一变结果就不一样了，实验不可复现。
    """
    by_user = defaultdict(list)
    for u, i, t in inter:
        by_user[u].append((t, i))
    return {u: [i for _, i in sorted(v, key=lambda x: (x[0], x[1]))] for u, v in by_user.items()}


def reindex(seqs):
    """
    字符串 ID -> 连续整数。
    item 从 1 开始编号: 0 永久留给 PAD。后面 nn.Embedding(padding_idx=0)
    才对得上，这是序列推荐的通用约定，现在偷懒后面一定会踩。
    user 从 0 开始，因为 user id 暂时只做统计，不进模型。
    """
    item2idx, user2idx = {}, {}
    out = {}
    for u in sorted(seqs.keys()):          # sorted 保证可复现
        user2idx[u] = len(user2idx)
        idx_seq = []
        for asin in seqs[u]:
            if asin not in item2idx:
                item2idx[asin] = len(item2idx) + 1   # 从 1 开始
            idx_seq.append(item2idx[asin])
        out[user2idx[u]] = idx_seq
    return out, user2idx, item2idx


def load_meta(meta_path, item2idx):
    """
    只保留出现在交互里的 item 的元信息。
    categories 是嵌套 list（一个 item 可能属于多条类目路径），这里拍平成
    最长的一条路径 —— S1 拼文本时用它，S4 验证 c_1 语义有效性时也用它。
    """
    meta = {}
    for m in parse_gz(meta_path):
        asin = m.get("asin")
        if asin not in item2idx:
            continue
        cats = m.get("categories", [])
        cat_path = max(cats, key=len) if cats else []
        meta[item2idx[asin]] = {
            "asin": asin,
            "title": m.get("title", "") or "",
            "brand": m.get("brand", "") or "",
            "price": m.get("price", None),
            "categories": cat_path,
            "description": (m.get("description", "") or "")[:500],
        }
    return meta


def main(name="beauty", review_file="reviews_Beauty_5.json.gz",
         meta_file="meta_Beauty.json.gz"):
    out_dir = os.path.join(OUT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    inter = load_interactions(os.path.join(RAW_DIR, review_file))
    print(f"[load] raw interactions: {len(inter)}")

    inter = dedup(inter)
    print(f"[dedup] after dedup: {len(inter)}")

    inter = k_core(inter, k=5)
    print(f"[k-core] after filtering: {len(inter)}")

    seqs = build_sequences(inter)
    seqs, user2idx, item2idx = reindex(seqs)

    meta = load_meta(os.path.join(RAW_DIR, meta_file), item2idx)

    # 元信息覆盖率: 缺 title 的 item 在 S1 会拿到空文本 -> 语义 embedding 无意义。
    # 这个数字必须现在就知道，它直接决定 Semantic ID 的质量上限。
    n_items = len(item2idx)
    n_with_title = sum(1 for v in meta.values() if v["title"].strip())
    lens = [len(s) for s in seqs.values()]

    stats = {
        "dataset": name,
        "num_users": len(seqs),
        "num_items": n_items,
        "num_interactions": sum(lens),
        "avg_seq_len": round(sum(lens) / len(lens), 2),
        "min_seq_len": min(lens),
        "max_seq_len": max(lens),
        "meta_coverage": round(len(meta) / n_items, 4),
        "title_coverage": round(n_with_title / n_items, 4),
    }
    print(json.dumps(stats, indent=2))

    # 只存整段序列，不在这里切 train/val/test。
    # 原因: leave-one-out 的切法是 seq[:-2] / seq[-2] / seq[-1]，一行代码的事，
    # 放到 Dataset 里做。存三份文件会导致三份数据不同步，是常见的隐性 bug 源。
    with open(f"{out_dir}/sequences.json", "w") as f:
        json.dump(seqs, f)
    with open(f"{out_dir}/item_meta.json", "w") as f:
        json.dump(meta, f)
    with open(f"{out_dir}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[save] wrote to {out_dir}")


if __name__ == "__main__":
    main()