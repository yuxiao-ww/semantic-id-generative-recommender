"""
S1 验收: 随机抽 item，看它的最近邻是不是语义相似。
如果最近邻在语义上八竿子打不着，说明 embedding 有问题（大概率是
pooling 用错、或忘了归一化），此时绝对不能进 S3。
"""

import json
import numpy as np
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

name = "beauty"
data_dir = f"{PROJECT_ROOT}/data/processed/{name}"
emb = np.load(f"{data_dir}/item_emb.npy")
meta = json.load(open(f"{data_dir}/item_meta.json"))

# 已归一化，内积 == 余弦相似度
sims = emb @ emb.T          # (N+1, N+1)

rng = np.random.default_rng(42)
probes = rng.choice(range(1, emb.shape[0]), size=5, replace=False)

for i in probes:
    row = sims[i].copy()
    row[i] = -1             # 排除自己
    row[0] = -1             # 排除 PAD
    top = np.argsort(-row)[:5]
    print(f"\n=== Query item {i}: {meta[str(i)]['title'][:60]} ===")
    for j in top:
        print(f"  sim={sims[i][j]:.3f}  {meta[str(j)]['title'][:60]}")
