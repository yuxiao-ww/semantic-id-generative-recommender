"""
S5a: GR 的 SID token 序列数据集

把 item 序列翻译成带层偏移的 SID token 序列，做 next-token prediction。

token 布局 (方案B):
  第 l 层 (l=0..3) 的码字 k -> token id = k + l*256
  BOS = 4*256 = 1024,  PAD = 1025,  vocab = 1026

leave-one-out 切分和 S2a 完全一致:
  train: seq[:-2] 内部做 next-token
  val:   输入 seq[:-2] 的SID，预测 seq[-2] 的SID
  test:  输入 seq[:-1] 的SID，预测 seq[-1] 的SID
"""

import json
import os
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NUM_CODES = 256
N_LAYERS = 4
BOS = N_LAYERS * NUM_CODES        # 1024
PAD = BOS + 1                     # 1025
VOCAB_SIZE = PAD + 1              # 1026


def sid_to_tokens(sid):
    """
    一个 item 的 4 位 SID -> 4 个带偏移的 token id。
    sid = [k1,k2,k3,k4] -> [k1+0*256, k2+1*256, k3+2*256, k4+3*256]
    """
    return [sid[l] + l * NUM_CODES for l in range(N_LAYERS)]


def tokens_to_sid(tokens):
    """
    逆操作: 4 个 token id -> 4 位 SID (S6 生成后查表要用)。
    减去层偏移还原成 0~255 的码字。
    """
    return [tokens[l] - l * NUM_CODES for l in range(N_LAYERS)]


class GRDataset(Dataset):
    def __init__(self, split, max_items=50,
                 seq_file="sequences.json", sid_file="item_sid.json"):
        # 1. 序列文件和SID文件可配置: 冷启动实验用 warm序列 + cold SID
        #    默认值和S5一致，不影响之前的训练/评测
        data_dir = f"{PROJECT_ROOT}/data/processed/beauty"
        with open(f"{data_dir}/{seq_file}") as f:
            raw = json.load(f)
        self.seqs = [raw[k] for k in sorted(raw.keys(), key=int)]
        self.seqs = [s for s in self.seqs if len(s) >= 3]
        with open(f"{data_dir}/{sid_file}") as f:
            self.item_sid = json.load(f)

        self.split = split
        self.max_items = max_items            # 最多保留多少个历史 item
        self.max_tokens = max_items * N_LAYERS + 1   # +1 是 BOS

    def __len__(self):
        return len(self.seqs)

    def _items_to_token_seq(self, items):
        """
        item id 列表 -> 展平的 token 序列，前面加 BOS。
        [BOS, k1¹,k2¹,k3¹,k4¹, k1²,k2²,k3²,k4², ...]
        """
        tokens = [BOS]
        for it in items:
            tokens += sid_to_tokens(self.item_sid[str(it)])
        return tokens

    def _pad_left(self, tokens, length):
        """左 padding 到定长 (和 SASRec 一致，保证最新 item 在右端)。"""
        tokens = tokens[-length:]
        return [PAD] * (length - len(tokens)) + tokens

    def __getitem__(self, idx):
        seq = self.seqs[idx]

        if self.split == "train":
            # 1. 训练用 seq[:-2]，内部做 next-token prediction
            items = seq[:-2][-self.max_items:]
            tokens = self._items_to_token_seq(items)      # 含 BOS
            # 2. input = tokens[:-1], target = tokens[1:] 错位一位
            inp = tokens[:-1]
            tgt = tokens[1:]
            # 3. 左 padding 到定长；target 的 PAD 位置在 loss 里会被 mask
            inp = self._pad_left(inp, self.max_tokens)
            tgt = self._pad_left(tgt, self.max_tokens)
            return {
                "input": torch.tensor(inp, dtype=torch.long),
                "target": torch.tensor(tgt, dtype=torch.long),
            }

        # val / test: 输入历史 SID，答案是下一个 item 的 4 位 SID
        if self.split == "val":
            hist_items = seq[:-2][-self.max_items:]
            answer_item = seq[-2]
        else:  # test
            hist_items = seq[:-1][-self.max_items:]
            answer_item = seq[-1]

        # 1. 输入 = BOS + 历史SID (生成的起点)
        inp = self._items_to_token_seq(hist_items)
        inp = self._pad_left(inp, self.max_tokens)
        # 2. 答案 = 下一个 item 的 4 位 token (生成后和它比)
        answer_tokens = sid_to_tokens(self.item_sid[str(answer_item)])
        return {
            "input": torch.tensor(inp, dtype=torch.long),
            "answer_item": answer_item,                    # 用于评测查表
            "answer_tokens": torch.tensor(answer_tokens, dtype=torch.long),
        }


def train_collate(batch):
    return {
        "input": torch.stack([b["input"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
    }


def eval_collate(batch):
    return {
        "input": torch.stack([b["input"] for b in batch]),
        "answer_item": [b["answer_item"] for b in batch],
        "answer_tokens": torch.stack([b["answer_tokens"] for b in batch]),
    }