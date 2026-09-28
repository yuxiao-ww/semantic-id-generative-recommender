"""
S7c: 冷启动对决 —— GR (Trie约束解码) vs SASRec 在 cold test 上的 Recall@K

对843个cold test用户(历史全warm，答案是cold item):
  GR: Trie约束beam search，只在真实item的SID路径搜索，cold item可达
  SASRec: 全量排序，cold item随机embedding，打分≈随机 -> 几乎归零

这是整个项目的核心对比证据。
"""

import json
import os
import numpy as np
import torch

from gr.model import GRModel
from gr.dataset import GRDataset, sid_to_tokens, N_LAYERS, PAD, BOS
from gr.generate import build_sid_to_item, generate_recommendations
from gr.trie import SidTrie
from baselines.sasrec import SASRec

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_cold_test():
    data_dir = f"{PROJECT_ROOT}/data/processed/beauty"
    with open(f"{data_dir}/cold_test.json") as f:
        return json.load(f)


# ---------- GR 冷启动评测 ----------
def build_gr_input(history, item_sid, max_items=50):
    """把一个用户的warm历史转成GR的输入token序列(含BOS,左padding)。"""
    NUM_CODES = 256
    max_tokens = max_items * N_LAYERS + 1
    items = history[-max_items:]
    tokens = [BOS]
    for it in items:
        sid = item_sid[str(it)]
        tokens += [sid[l] + l * NUM_CODES for l in range(N_LAYERS)]
    tokens = tokens[-max_tokens:]
    tokens = [PAD] * (max_tokens - len(tokens)) + tokens
    return tokens


@torch.no_grad()
def eval_gr_cold(cold_test, item_sid, sid2item, trie, device,
                 beam_width=50, ks=(5, 10, 20)):
    with open(f"{PROJECT_ROOT}/data/processed/beauty/cold_items.json") as f:
        cold_set = set(json.load(f))
    model = GRModel(max_len=50 * N_LAYERS + 1, d_model=128,
                    n_heads=4, n_blocks=4).to(device)
    model.load_state_dict(torch.load(f"{PROJECT_ROOT}/checkpoints/gr_warm.pt",
                                     map_location=device))
    model.eval()

    recalls = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}
    max_k = max(ks)

    batch_size = 128
    for start in range(0, len(cold_test), batch_size):
        chunk = cold_test[start:start + batch_size]
        inputs = torch.tensor(
            [build_gr_input(s["history"], item_sid) for s in chunk],
            dtype=torch.long).to(device)
        # Trie约束解码: 只在真实item路径搜索，cold item可达
        rec_lists, _, _ = generate_recommendations(
            model, inputs, beam_width, sid2item, device, trie=trie)
        for i, s in enumerate(chunk):
            answer = s["answer"]
            recs = rec_lists[i]
            # ===== 临时诊断 =====
            if start == 0 and i < 5:
                # 生成的recs里有几个是cold item?
                n_cold_in_recs = sum(1 for r in recs if r in cold_set)
                print(f"  [debug] user{i}: answer={answer}(cold), "
                      f"recs前5={recs[:5]}, "
                      f"recs里cold item数={n_cold_in_recs}/{len(recs)}, "
                      f"answer在recs? {answer in recs}")
            # ====================
            rank = recs.index(answer) + 1 if answer in recs else max_k + 1
            for k in ks:
                recalls[k].append(1.0 if rank <= k else 0.0)
                ndcgs[k].append((1.0 / np.log2(rank + 1)) if rank <= k else 0.0)
        print(f"  [GR] {min(start+batch_size, len(cold_test))}/{len(cold_test)} done")

    return {f"Recall@{k}": np.mean(recalls[k]) for k in ks}, \
           {f"NDCG@{k}": np.mean(ndcgs[k]) for k in ks}


# ---------- SASRec 冷启动评测 ----------
def build_sasrec_input(history, max_len=50):
    """warm历史转SASRec输入(item id序列,左padding)。"""
    seq = history[-max_len:]
    return [0] * (max_len - len(seq)) + seq


@torch.no_grad()
def eval_sasrec_cold_retrieval(cold_test, num_items, cold_items, device, ks=(5,10,20)):
    """SASRec检索式: 只在cold item候选里排序 (和GR同候选集，公平对比)。"""
    model = SASRec(num_items=num_items, max_len=50, d_model=64,
                   n_heads=1, n_blocks=2, dropout=0.5).to(device)
    model.load_state_dict(torch.load(f"{PROJECT_ROOT}/checkpoints/sasrec_warm.pt",
                                     map_location=device))
    model.eval()

    cold_tensor = torch.tensor(sorted(cold_items), device=device)  # cold item ids
    recalls = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}

    bs = 256
    for start in range(0, len(cold_test), bs):
        chunk = cold_test[start:start+bs]
        inp = torch.tensor([build_sasrec_input(s["history"]) for s in chunk],
                           dtype=torch.long, device=device)
        answers = torch.tensor([s["answer"] for s in chunk], device=device)
        scores = model.score_all(inp)                  # (B, num_items+1)
        # 只保留cold item列的分数，其余设-inf (候选集限定cold)
        mask = torch.full_like(scores, float("-inf"))
        mask[:, cold_tensor] = scores[:, cold_tensor]
        scores = mask
        ans_scores = scores[torch.arange(len(chunk)), answers]
        rank = (scores > ans_scores.unsqueeze(1)).sum(1) + 1
        rank = rank.cpu().numpy()
        for k in ks:
            recalls[k].extend((rank <= k).astype(float).tolist())
            ndcgs[k].extend(np.where(rank<=k, 1.0/np.log2(rank+1), 0.0).tolist())

    return {f"Recall@{k}": np.mean(recalls[k]) for k in ks}, \
           {f"NDCG@{k}": np.mean(ndcgs[k]) for k in ks}


@torch.no_grad()
def eval_gr_cold_retrieval(cold_test, item_sid, cold_items, device, ks=(5, 10, 20)):
    """
    检索式冷启动评测: 候选集=全部cold item，GR对每个cold item计算其SID的
    生成log概率，按概率排序算Recall。测的是GR语义表示的冷启动排序能力。
    """
    NUM_CODES = 256
    model = GRModel(max_len=50 * N_LAYERS + 1, d_model=128,
                    n_heads=4, n_blocks=4).to(device)
    model.load_state_dict(torch.load(f"{PROJECT_ROOT}/checkpoints/gr_warm.pt",
                                     map_location=device))
    model.eval()

    # 1. 预计算全部cold item的SID token序列 (C, 4)
    cold_list = sorted(cold_items)
    cold_sid_tokens = torch.tensor(
        [sid_to_tokens(item_sid[str(it)]) for it in cold_list],
        dtype=torch.long, device=device)                  # (C, 4)
    C = len(cold_list)
    # cold item id -> 在cold_list中的下标 (用于定位answer)
    cold_pos = {it: idx for idx, it in enumerate(cold_list)}

    recalls = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}
    max_k = max(ks)

    for ui, s in enumerate(cold_test):
        answer = s["answer"]
        # 2. 构造该用户输入 (含BOS + warm历史SID, 左padding)
        # 构造输入: BOS + warm历史SID，但裁剪到留出 N_LAYERS-1 个拼接位
        #    max_len - (N_LAYERS-1) 是历史能占的最大长度，防止拼candidate后越界
        max_hist_tokens = model.max_len - (N_LAYERS - 1)  # 201 - 3 = 198
        hist_tokens = [BOS]
        for it in s["history"][-50:]:
            sid = item_sid[str(it)]
            hist_tokens += [sid[l] + l * NUM_CODES for l in range(N_LAYERS)]
        # 只保留最近的，且不超过 max_hist_tokens
        hist_tokens = hist_tokens[-max_hist_tokens:]
        inp = torch.tensor([hist_tokens], dtype=torch.long, device=device)  # (1, L)

        # 3. teacher-forcing计算每个cold item SID的生成log概率
        #    做法: 把输入序列 + 每个cold item的4个token拼接，一次forward拿各层概率
        #    为效率，分批处理cold item
        scores = torch.empty(C, device=device)
        batch = 512
        for b0 in range(0, C, batch):
            cand = cold_sid_tokens[b0:b0 + batch]          # (b, 4)
            bsz = cand.size(0)
            # 3.1 输入复制bsz份，尾部接上candidate的前3个token(第4个只需概率不需输入)
            base = inp.expand(bsz, -1)                     # (b, L)
            # 拼接: [历史..., k1, k2, k3] -> 预测k1,k2,k3,k4的位置
            seq = torch.cat([base, cand[:, :N_LAYERS - 1]], dim=1)  # (b, L+3)
            logits = model(seq)                            # (b, L+3, V)
            # 3.2 取最后4个位置的logits: 分别预测第1,2,3,4层token
            #     位置 L-1 预测k1, L 预测k2, L+1 预测k3, L+2 预测k4
            last4 = logits[:, -N_LAYERS:, :]               # (b, 4, V)
            logp = torch.log_softmax(last4, dim=-1)        # (b, 4, V)
            # 3.3 取每层对应candidate token的log概率，求和=该SID的生成log概率
            tok = cand.unsqueeze(-1)                        # (b, 4, 1)
            step_logp = logp.gather(2, tok).squeeze(-1)    # (b, 4)
            scores[b0:b0 + bsz] = step_logp.sum(dim=1)     # (b,)

        # 4. answer在cold候选里的rank
        ans_score = scores[cold_pos[answer]]
        rank = (scores > ans_score).sum().item() + 1
        for k in ks:
            recalls[k].append(1.0 if rank <= k else 0.0)
            ndcgs[k].append((1.0 / np.log2(rank + 1)) if rank <= k else 0.0)

        if (ui + 1) % 200 == 0:
            print(f"  [GR-retrieval] {ui+1}/{len(cold_test)} done")

    return {f"Recall@{k}": np.mean(recalls[k]) for k in ks}, \
           {f"NDCG@{k}": np.mean(ndcgs[k]) for k in ks}


def main():
    device = get_device()
    data_dir = f"{PROJECT_ROOT}/data/processed/beauty"

    cold_test = load_cold_test()
    print(f"[cold-eval] {len(cold_test)} cold-start test users")

    # 1. 加载所需数据
    with open(f"{data_dir}/item_sid_cold.json") as f:
        item_sid = json.load(f)                    # item -> 4位SID
    with open(f"{data_dir}/cold_items.json") as f:
        cold_items = set(json.load(f))             # cold item id集合
    num_items = np.load(f"{data_dir}/item_emb.npy").shape[0] - 1

    # 2. 检索式冷启动评测 (候选集=全部cold item)
    print("\n[GR] retrieval-based cold scoring...")
    gr_r, gr_n = eval_gr_cold_retrieval(cold_test, item_sid, cold_items, device)

    print("[SASRec] retrieval-based cold scoring...")
    sr_r, sr_n = eval_sasrec_cold_retrieval(cold_test, num_items, cold_items, device)

    # 3. 结果对比
    print("\n" + "=" * 60)
    print("COLD-START RESULTS (retrieval over cold-item candidates)")
    print("=" * 60)
    print(f"{'Metric':<14}{'GR (semantic)':<18}{'SASRec (random emb)':<20}")
    for k in (5, 10, 20):
        print(f"Recall@{k:<8}{gr_r[f'Recall@{k}']:<18.4f}{sr_r[f'Recall@{k}']:<20.4f}")
        print(f"NDCG@{k:<10}{gr_n[f'NDCG@{k}']:<18.4f}{sr_n[f'NDCG@{k}']:<20.4f}")
    print("=" * 60)
    # 随机baseline参考: 候选集大小
    print(f"(random baseline Recall@10 ≈ 10/{len(cold_items)} = "
          f"{10/len(cold_items):.4f})")


if __name__ == "__main__":
    main()