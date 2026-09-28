"""
S5b: 生成式推荐 Transformer (GR)

结构和 SASRec decoder 类似 (causal self-attention + pre-LN block)，
两处关键区别:
  1. 输出层预测 token (1026 词表)，不是 item
  2. 训练用 full softmax + 交叉熵，不需要负采样 (词表只有1026，softmax负担得起)

token 布局见 gr/dataset.py: 4*256 SID tokens + BOS(1024) + PAD(1025)
"""

import torch
import torch.nn as nn

from gr.dataset import VOCAB_SIZE, PAD


class GRModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, max_len=201, d_model=128,
                 n_heads=4, n_blocks=4, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model

        # 1. token embedding: 覆盖所有 4 层 token + BOS + PAD 的统一表
        #    padding_idx=PAD 让 PAD token 的 embedding 梯度为 0
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        # 2. 可学习位置 embedding
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.dropout = nn.Dropout(dropout)
        self.emb_ln = nn.LayerNorm(d_model)

        # 3. n_blocks 个 pre-LN Transformer block (和 SASRec 同款结构)
        self.attn_lns = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_blocks)])
        self.attns = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_blocks)
        ])
        self.ffn_lns = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_blocks)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(d_model * 4, d_model),
            ) for _ in range(n_blocks)
        ])
        self.n_blocks = n_blocks

        # 4. 输出头: hidden state -> 1026 维 token logits (这是和 SASRec 的核心区别)
        self.out_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.tok_emb.weight[PAD].fill_(0)

    def _causal_mask(self, L, device):
        # 上三角 (不含对角) 为 True = 屏蔽未来 (和 SASRec 一致)
        return torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()

    def forward(self, input_ids):
        """
        input_ids: (B, L) token 序列
        return: (B, L, vocab_size) 每个位置对下一个 token 的 logits
        """
        B, L = input_ids.shape
        device = input_ids.device

        # 防御: 序列长度不能超过位置编码上限，否则 pos_emb 越界
        assert L <= self.max_len, f"seq len {L} > max_len {self.max_len}"

        # 1. token embedding + 位置 embedding
        x = self.tok_emb(input_ids)
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        x = x + self.pos_emb(positions)
        x = self.emb_ln(self.dropout(x))

        # 2. mask: causal (屏蔽未来) + padding (屏蔽 PAD token)
        pad_mask = (input_ids == PAD)              # (B, L)
        causal = self._causal_mask(L, device)      # (L, L)

        # 3. pre-LN blocks
        for i in range(self.n_blocks):
            h = self.attn_lns[i](x)
            attn_out, _ = self.attns[i](
                h, h, h, attn_mask=causal,
                key_padding_mask=pad_mask, need_weights=False,
            )
            # 1. 修复 padding 导致的 NaN:完全由PAD组成的query行,其attention输出是
            #    softmax(全-inf)=NaN。这些位置本就是padding,输出无意义,直接清零,
            #    阻断NaN通过残差扩散到有效位置。
            attn_out = torch.nan_to_num(attn_out, nan=0.0)
            x = x + attn_out
            h = self.ffn_lns[i](x)
            x = x + self.ffns[i](h)

        # 4. 输出头: 每个位置预测下一个 token
        x = self.out_ln(x)
        logits = self.head(x)                      # (B, L, vocab_size)
        return logits