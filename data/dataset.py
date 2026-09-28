"""
S2a: 序列推荐的 Dataset / DataLoader

三种 split 共用一份 sequences.json，切分点不同:
  train: 用 seq[:-2] 内部所有位置做 next-item 监督
  val:   输入 seq[:-2]，预测 seq[-2]
  test:  输入 seq[:-1]，预测 seq[-1]

关键约定 (和 S0 对齐):
  - item 从 1 编号，0 是 PAD
  - 序列左侧 padding: 把序列贴到定长张量的右端，最新的 item 永远在最右边，
    这样"最后一个位置的 hidden state"稳定对应"最近一次交互"，是 SASRec 的标准做法
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class SeqRecDataset(Dataset):
    # def __init__(self, data_dir, split, max_len=50):
        # 1. 读全量序列，key 是字符串，转成 int 排序保证可复现
        # with open(os.path.join(data_dir, "sequences.json")) as f:
        #     raw = json.load(f)
        # 1. 序列文件可配置: 冷启动实验用 sequences_warm.json
        #    默认值不变，不影响S2的调用
    def __init__(self, data_dir, split, max_len=50, seq_file="sequences.json"):
        with open(os.path.join(data_dir, seq_file)) as f:
            raw = json.load(f)
        self.seqs = [raw[k] for k in sorted(raw.keys(), key=int)]

        # 2. 只保留长度 >=3 的用户: 至少要能切出 train(1)+val(1)+test(1)
        self.seqs = [s for s in self.seqs if len(s) >= 3]

        self.split = split
        self.max_len = max_len

        # 3. num_items 从全量数据统计，三种 split 必须一致，否则评测排序维度对不上
        # self.num_items = max(max(s) for s in self.seqs)
        # 3. num_items: warm序列缺cold item，会低估。冷启动实验必须传入真实总数，
        #    保证embedding表覆盖全部item(含cold)，cold item才有(随机)embedding槽位
        self.num_items = max(max(s) for s in self.seqs)
        self._forced_num_items = None  # 训练脚本可覆盖

    def set_num_items(self, n):
        """冷启动: 强制num_items为全量item数(含cold)。"""
        self.num_items = n

    def __len__(self):
        return len(self.seqs)

    def _pad(self, seq):
        """
        左 padding 到 max_len。
        seq 太长: 只留最近 max_len 个 (推荐系统里近期行为信息量最大)。
        seq 太短: 左边补 0。
        """
        seq = seq[-self.max_len:]
        pad_len = self.max_len - len(seq)
        return [0] * pad_len + seq

    def __getitem__(self, idx):
        seq = self.seqs[idx]

        if self.split == "train":
            # 1. 训练序列去掉最后两个 (它们分别是 val/test 的答案，不能泄漏)
            train_seq = seq[:-2]
            # 2. 输入是 train_seq[:-1]，目标是 train_seq[1:]，整条错位一位
            #    即每个位置都预测它的下一个 item —— next-item prediction
            inp = train_seq[:-1]
            tgt = train_seq[1:]
            inp = self._pad(inp)
            # 3. target 也左 padding，pad 位置填 0，loss 里会 mask 掉
            tgt = [0] * (self.max_len - len(tgt[-self.max_len:])) + tgt[-self.max_len:]
            return {
                "input": torch.tensor(inp, dtype=torch.long),
                "target": torch.tensor(tgt, dtype=torch.long),
            }

        elif self.split == "val":
            # 1. 输入 seq[:-2]，答案是 seq[-2]
            inp = self._pad(seq[:-2])
            answer = seq[-2]
        else:  # test
            # 1. 输入 seq[:-1]，答案是 seq[-1]
            inp = self._pad(seq[:-1])
            answer = seq[-1]

        # 2. 评测时还要知道这个用户"见过哪些 item"，评测排序时把它们 mask 掉
        #    (不能把用户已经交互过的 item 推荐给他，也避免它们挤占 top-K 名次)
        seen = set(seq)
        return {
            "input": torch.tensor(inp, dtype=torch.long),
            "answer": torch.tensor(answer, dtype=torch.long),
            "seen": seen,   # set，collate 时特殊处理
        }


def eval_collate(batch):
    """
    val/test 的 collate。
    seen 是变长 set，默认 collate 会报错，这里手动打包成 list。
    """
    return {
        "input": torch.stack([b["input"] for b in batch]),
        "answer": torch.stack([b["answer"] for b in batch]),
        "seen": [b["seen"] for b in batch],
    }


from data.dataset import SeqRecDataset, eval_collate
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ds = SeqRecDataset(f"{PROJECT_ROOT}/data/processed/beauty", split="test", max_len=50)
print("num_users:", len(ds), "num_items:", ds.num_items)

loader = DataLoader(ds, batch_size=4, collate_fn=eval_collate)
b = next(iter(loader))
print("input:", b["input"].shape)      # 期望 (4, 50)
print("answer:", b["answer"].shape)    # 期望 (4,)
print("seen[0] size:", len(b["seen"][0]))
print("sample input[0]:", b["input"][0])   # 应该看到左边一堆 0，右边是真实 item
print(b["answer"][0])