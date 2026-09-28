"""
S7a: 冷启动数据准备

1. 选最冷门10%的item作为cold set(模拟新上架商品)
2. 从训练序列删除cold item(保证模型训练时对cold item完全不可见)
3. 构造cold test: 答案是cold item的用户，用来测冷启动推荐能力

产出:
  cold_items.json       cold item id列表
  sequences_warm.json    删除cold item后的序列(重训用)
  cold_test.json         冷启动测试样本 [{history:[...], answer: cold_item}]
"""

import json
import os
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(cold_ratio=0.1):
    data_dir = f"{PROJECT_ROOT}/data/processed/beauty"
    with open(f"{data_dir}/sequences.json") as f:
        seqs = {int(k): v for k, v in json.load(f).items()}

    # 1. 统计每个item的交互次数(在所有序列里出现多少次)
    item_freq = Counter()
    for uid, seq in seqs.items():
        for it in seq:
            item_freq[it] += 1

    # 2. 按频次升序，取最冷门的10%作为cold set
    #    冷门item当"新品"，既符合冷启动真实场景，又对训练集破坏小
    all_items = sorted(item_freq.keys(), key=lambda x: item_freq[x])
    n_cold = int(len(all_items) * cold_ratio)
    cold_items = set(all_items[:n_cold])
    print(f"[cold] total items: {len(all_items)}, "
          f"cold items (bottom {int(cold_ratio*100)}%): {len(cold_items)}")
    print(f"[cold] cold item freq range: "
          f"{item_freq[all_items[0]]} ~ {item_freq[all_items[n_cold-1]]}")

    # 3. 构造 warm 训练序列: 从每个用户序列删除 cold item
    #    注意: 用户的最后一个item(test答案)如果是cold，这个用户会进cold_test，
    #    但他的历史部分仍要删掉其中的cold item
    warm_seqs = {}
    cold_test = []
    for uid, seq in seqs.items():
        # 3.1 答案(最后一个item)是不是cold?
        answer = seq[-1]
        history = seq[:-1]
        # 3.2 历史里删掉所有cold item(模型训练/输入都不该看到cold)
        warm_history = [it for it in history if it not in cold_items]

        if answer in cold_items:
            # 3.3 答案是cold -> 这是一个冷启动测试样本
            #     需要历史至少还剩几个warm item才有意义(否则没上下文)
            if len(warm_history) >= 2:
                cold_test.append({
                    "history": warm_history,
                    "answer": answer,
                })
        else:
            # 3.4 答案是warm -> 这个用户的(去cold)序列用于重训
            #     warm_history + warm answer 构成干净的warm序列
            full_warm = warm_history + [answer]
            if len(full_warm) >= 3:      # 保持和之前一样的最小长度要求
                warm_seqs[uid] = full_warm

    print(f"[cold] warm training sequences: {len(warm_seqs)}")
    print(f"[cold] cold-start test samples: {len(cold_test)}")

    # 4. 保存
    os.makedirs(f"{data_dir}", exist_ok=True)
    with open(f"{data_dir}/cold_items.json", "w") as f:
        json.dump(sorted(cold_items), f)
    with open(f"{data_dir}/sequences_warm.json", "w") as f:
        json.dump({str(k): v for k, v in warm_seqs.items()}, f)
    with open(f"{data_dir}/cold_test.json", "w") as f:
        json.dump(cold_test, f)
    print(f"[save] cold_items.json / sequences_warm.json / cold_test.json")


if __name__ == "__main__":
    main()