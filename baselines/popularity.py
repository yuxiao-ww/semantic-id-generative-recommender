"""
S2c-1: Popularity baseline

作用: 验证 S2b 评测器 + 提供性能地板。
逻辑: 统计训练集 item 频次，对所有用户输出同一套分数 (频次)。
不看用户历史顺序，是最弱的非平凡模型。

实现评测器要求的接口: score_all(input) -> (B, num_items+1)
"""

import os
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PopularityModel:
    def __init__(self, num_items, device):
        self.num_items = num_items
        self.device = device
        # 1. pop[i] = item i 的训练集出现次数，先全 0，fit 时填充
        self.pop = torch.zeros(num_items + 1, device=device)

    def fit(self, train_dataset):
        """
        统计频次。
        注意只能用训练集: 每个用户的 seq[:-2] 部分，绝不能碰 val/test 的 item，
        否则就是数据泄漏 (把答案偷看进了统计量)。
        """
        # 1. 遍历所有用户，只取 seq[:-2] (和 S2a train split 一致)
        for seq in train_dataset.seqs:
            for it in seq[:-2]:
                self.pop[it] += 1
        # 2. PAD 永远不推荐
        self.pop[0] = 0.0
        print(f"[popularity] fitted, max freq={int(self.pop.max())}")

    def eval(self):
        """空实现: 评测器会调 model.eval()，Popularity 没有 train/eval 模式之分。"""
        pass

    @torch.no_grad()
    def score_all(self, input_batch):
        """
        对 batch 里每个用户都返回同一套分数 (频次)。
        input_batch: (B, L)，这里其实不看它的内容，只用它的 batch size，
                     因为 Popularity 对谁都推一样的东西。
        返回: (B, num_items+1)
        """
        B = input_batch.size(0)
        # 1. 把 (num_items+1,) 的频次向量广播成 (B, num_items+1)
        return self.pop.unsqueeze(0).expand(B, -1).clone()
