"""
S2c-2: SASRec 模型结构

decoder-only Transformer 做序列推荐:
  item_emb + pos_emb -> N x [causal self-attention + FFN] -> 每个位置预测下一个 item

关键设计:
  - causal mask: 位置 t 只能看 <=t，避免用未来预测未来
  - item embedding 输入/输出共享: 参数减半，输入输出空间对齐
  - score_all(): 取最后位置 hidden state 和全量 item embedding 内积打分
    (实现 S2b 评测器要求的接口契约)
"""

import numpy as np
import torch
import torch.nn as nn


class SASRec(nn.Module):
    def __init__(self, num_items, max_len=50, d_model=64,
                 n_heads=1, n_blocks=2, dropout=0.5):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.d_model = d_model

        # 1. item embedding 表: num_items+1 行，第 0 行是 PAD，padding_idx=0
        #    梯度永久为 0，PAD 不会学到任何东西 (S0 里 item 从 1 编号就是为了这)
        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        # 2. 可学习位置 embedding: max_len 个位置，每个一个 d_model 向量
        #    SASRec 用 learned position 而非 sinusoidal (论文默认)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.dropout = nn.Dropout(dropout)
        self.emb_ln = nn.LayerNorm(d_model)   # embedding 后先过一层 LN，稳定训练

        # 3. 堆 n_blocks 个 Transformer block
        #    每个 block = 自注意力 + FFN，各自带残差 + LN
        self.attn_lns = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_blocks)])
        self.attns = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_blocks)
        ])
        self.ffn_lns = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_blocks)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(d_model, d_model),
            ) for _ in range(n_blocks)
        ])
        self.n_blocks = n_blocks

        self._init_weights()

    def _init_weights(self):
        # 1. embedding 用正态初始化，PAD 行强制清零
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].fill_(0)

    def _causal_mask(self, L, device):
        """
        生成 (L, L) 的下三角 causal mask。
        MultiheadAttention 约定: mask 为 True 的位置会被屏蔽 (置 -inf)。
        我们要屏蔽"未来" -> 上三角 (不含对角线) 为 True。
        """
        # 1. triu(diagonal=1): 严格上三角为 1，即 s>t 的位置 (未来)
        mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        return mask

    def _encode(self, input_ids):
        """
        序列 -> 每个位置的 hidden state。
        input_ids: (B, L)
        return: (B, L, d_model)
        """
        B, L = input_ids.shape
        device = input_ids.device

        # 1. item embedding 查表
        x = self.item_emb(input_ids)                 # (B, L, d)
        # 2. 加位置 embedding: 位置 0..L-1
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        x = x + self.pos_emb(positions)
        x = self.emb_ln(self.dropout(x))

        # 3. padding mask: PAD 位置 (input==0) 不参与 attention 的 key
        #    MultiheadAttention 的 key_padding_mask: True 表示屏蔽
        pad_mask = (input_ids == 0)                  # (B, L)
        causal = self._causal_mask(L, device)        # (L, L)

        # 4. 逐个 block: 都是 pre-LN 结构 (先 LN 再进子层，残差加回)
        for i in range(self.n_blocks):
            # 4.1 self-attention 子层
            h = self.attn_lns[i](x)
            attn_out, _ = self.attns[i](
                h, h, h,
                attn_mask=causal,                    # 屏蔽未来
                key_padding_mask=pad_mask,           # 屏蔽 PAD
                need_weights=False,
            )
            x = x + attn_out                         # 残差
            # 4.2 FFN 子层
            h = self.ffn_lns[i](x)
            x = x + self.ffns[i](h)                  # 残差

        return x                                     # (B, L, d)

    def forward(self, input_ids):
        """训练用: 返回每个位置的 hidden state，train.py 会拿它算 next-item loss。"""
        return self._encode(input_ids)

    @torch.no_grad()
    def score_all(self, input_ids):
        """
        评测接口 (S2b 契约): 对所有 item 打分。
        取序列最后一个位置的 hidden state (代表用户当前兴趣)，
        和全量 item embedding 做内积。
        return: (B, num_items+1)
        """
        h = self._encode(input_ids)                  # (B, L, d)
        # 1. 取最后位置: 左 padding 保证真实最后一个 item 落在 index -1
        last = h[:, -1, :]                           # (B, d)
        # 2. 和整张 item embedding 表内积 -> 每个 item 的分数
        scores = last @ self.item_emb.weight.t()     # (B, num_items+1)
        return scores


import os, torch
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # 放根目录跑

from baselines.sasrec import SASRec

model = SASRec(num_items=12101, max_len=50, d_model=64)
dummy = torch.randint(0, 12101, (4, 50))   # 假 batch
dummy[:, :30] = 0                          # 前 30 个当 PAD

h = model(dummy)
print("hidden:", h.shape)                  # 期望 (4, 50, 64)

scores = model.score_all(dummy)
print("scores:", scores.shape)             # 期望 (4, 12102)
print("PAD col masked in score_all?", scores[:, 0].tolist())  # 看 index 0 分数