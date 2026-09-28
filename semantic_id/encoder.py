"""
Stage 1: item 文本 -> bge-small CLS embedding

产出:
  data/processed/beauty/item_emb.npy   shape (num_items+1, 384), float32
  第 0 行是全 0 (PAD 占位)

设计要点:
  - 手写 tokenize/forward/pooling，不用 SentenceTransformer 封装（面试要能手撕）
  - CLS pooling，不是 mean（bge 官方用法）
  - L2 归一化：后面 RQ-VAE 用欧氏距离找最近码字，归一化后欧氏距离和
    余弦相似度单调等价，量化才有语义意义
"""

import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
MAX_LEN = 256          # bge 上限 512，item 文本拼完通常 100-200 token，256 够且省显存
BATCH_SIZE = 128


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_text(meta):
    """
    把 item 的结构化字段拼成一句话喂给编码器。

    字段顺序不是随便排的: title 放最前面，因为它信息密度最高，即使后面
    被 truncation 截断，最重要的信息也保住了。description 放最后且截断到
    200 字符 —— 它最长、噪声最多、边际价值最低。

    category 用 " > " 连接成路径 (如 "Beauty > Makeup > Lipstick")，
    保留层次信息，这对 RQ-VAE 学出层次化码字有帮助。
    """
    title = meta.get("title", "").strip()
    brand = meta.get("brand", "").strip()
    cats = meta.get("categories", [])
    cat_str = " > ".join(cats) if cats else ""
    price = meta.get("price", None)
    price_str = f"${price}" if price else ""
    desc = meta.get("description", "").strip()[:200]

    # 只拼非空字段，避免 "Brand:  Price: " 这种空洞噪声
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if cat_str:
        parts.append(f"Category: {cat_str}")
    if brand:
        parts.append(f"Brand: {brand}")
    if price_str:
        parts.append(f"Price: {price_str}")
    if desc:
        parts.append(f"Description: {desc}")
    text = ". ".join(parts)
    # 极端情况: 元信息全缺。给个占位符，别让 tokenizer 收到空串。
    return text if text else "Unknown product"


@torch.no_grad()
def encode_batch(texts, tokenizer, model, device):
    """
    一个 batch 的编码。三步:
      1. tokenize -> input_ids / attention_mask
      2. forward -> last_hidden_state  (B, L, 384)
      3. CLS pooling: 取每条序列第 0 个 token 的向量 -> (B, 384)
    """
    enc = tokenizer(
        texts, padding=True, truncation=True,
        max_length=MAX_LEN, return_tensors="pt",
    ).to(device)
    out = model(**enc)
    # bge 用 CLS pooling: [:, 0] 就是 [CLS] token 的 hidden state
    cls = out.last_hidden_state[:, 0]          # (B, 384)
    cls = torch.nn.functional.normalize(cls, p=2, dim=1)   # L2 归一化
    return cls.cpu().numpy()


def main(name="beauty"):
    data_dir = os.path.join(PROJECT_ROOT, "data/processed", name)
    with open(f"{data_dir}/item_meta.json") as f:
        meta = json.load(f)

    num_items = max(int(k) for k in meta.keys())   # item 从 1 编号
    device = get_device()
    print(f"[encode] device={device}, num_items={num_items}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    # 探测输出维度，别硬编码 384 —— 换 bge-base 时这里自动适配（S7 消融要用）
    probe = encode_batch(["probe"], tokenizer, model, device)
    dim = probe.shape[1]

    emb = np.zeros((num_items + 1, dim), dtype=np.float32)   # 第 0 行留给 PAD

    # 按 item index 顺序编码，保证 emb[i] 严格对应 item i
    ids = sorted(int(k) for k in meta.keys())
    for start in range(0, len(ids), BATCH_SIZE):
        batch_ids = ids[start:start + BATCH_SIZE]
        texts = [build_text(meta[str(i)]) for i in batch_ids]
        vecs = encode_batch(texts, tokenizer, model, device)
        for idx, i in enumerate(batch_ids):
            emb[i] = vecs[idx]
        if start % (BATCH_SIZE * 20) == 0:
            print(f"[encode] {start}/{len(ids)}")

    out_path = f"{data_dir}/item_emb.npy"
    np.save(out_path, emb)
    print(f"[save] {out_path}, shape={emb.shape}")


if __name__ == "__main__":
    main()
