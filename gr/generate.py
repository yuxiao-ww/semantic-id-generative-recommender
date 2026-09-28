"""
S6-1: GR 的 beam search 生成

对每个用户历史，autoregressive 生成 top-B 个下一个item的SID。
逐层生成(4步)，每步:
  - 只允许生成当前层区间的token(层约束)
  - 保留累积log概率最高的B个beam
生成后查 sid->item 表，过滤invalid，返回每个用户的 top-K item列表。

产出评测所需的推荐列表 + invalid rate 统计。
"""

import json
import os
import torch
import torch.nn.functional as F

from gr.dataset import NUM_CODES, N_LAYERS, BOS, PAD, tokens_to_sid

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_sid_to_item(item_sid_path):
    """
    建 SID元组 -> item_idx 的反查表，生成后用它把SID还原成item。
    key = (k1,k2,k3,k4) 四元组，value = item index。
    """
    with open(item_sid_path) as f:
        item_sid = json.load(f)
    sid2item = {}
    for item_idx, sid in item_sid.items():
        sid2item[tuple(sid)] = int(item_idx)
    return sid2item


@torch.no_grad()
def beam_search(model, input_ids, beam_width, device, trie=None):
    """
    beam search + 可选Trie约束解码。
    trie: SidTrie实例。传入则约束生成只走真实item的SID路径(invalid=0，cold可达);
          不传则退化为原来的层区间约束(向后兼容S6的warm评测)。
    """
    from gr.dataset import sid_to_tokens  # 顶部已import的话可删

    B = input_ids.size(0)
    device = input_ids.device

    max_input_len = model.max_len - N_LAYERS
    if input_ids.size(1) > max_input_len:
        input_ids = input_ids[:, -max_input_len:]

    beam_scores = torch.zeros(B, beam_width, device=device)
    # [TRIE] 记录每个beam已生成的token前缀(用于查Trie)，(B, beam_width, 已生成层数)
    gen_tokens = [[[] for _ in range(beam_width)] for _ in range(B)]
    first_step = True
    beams = input_ids.unsqueeze(1).expand(B, beam_width, -1).contiguous()

    for layer in range(N_LAYERS):
        lo = layer * NUM_CODES
        hi = lo + NUM_CODES

        flat = beams.view(B * beam_width, -1)
        logits = model(flat)
        next_logits = logits[:, -1, :]                # (B*bw, V)

        # [TRIE] 构造约束mask: 默认全-inf，只放开合法token
        mask = torch.full_like(next_logits, float("-inf"))
        if trie is None:
            # 无Trie: 退化为层区间约束(S6 warm评测兼容)
            mask[:, lo:hi] = 0.0
        else:
            # [TRIE] 逐beam查合法后继，只放开这些token
            for b in range(B):
                for w in range(beam_width):
                    prefix = gen_tokens[b][w]         # 已生成的前缀token
                    valid = trie.get_valid_next(prefix)   # set[int]合法后继
                    flat_idx = b * beam_width + w
                    for t in valid:
                        mask[flat_idx, t] = 0.0
        next_logits = next_logits + mask

        log_probs = F.log_softmax(next_logits, dim=-1)
        log_probs = log_probs.view(B, beam_width, -1)
        cand_scores = beam_scores.unsqueeze(-1) + log_probs

        if first_step:
            cand_scores0 = cand_scores[:, 0, :]
            top_scores, top_tokens = cand_scores0.topk(beam_width, dim=-1)
            beam_scores = top_scores
            new_tok = top_tokens.unsqueeze(-1)
            base = input_ids.unsqueeze(1).expand(B, beam_width, -1)
            beams = torch.cat([base, new_tok], dim=-1)
            # [TRIE] 记录第1层生成的token
            for b in range(B):
                for w in range(beam_width):
                    gen_tokens[b][w] = [top_tokens[b, w].item()]
            first_step = False
        else:
            flat_cand = cand_scores.view(B, -1)
            top_scores, top_idx = flat_cand.topk(beam_width, dim=-1)
            beam_id = top_idx // cand_scores.size(-1)
            token_id = top_idx % cand_scores.size(-1)
            beam_scores = top_scores
            new_beams = []
            # [TRIE] 按beam_id重排前缀记录，并追加新token
            new_gen = [[None]*beam_width for _ in range(B)]
            for b in range(B):
                parent = beams[b, beam_id[b]]
                child = torch.cat([parent, token_id[b].unsqueeze(-1)], dim=-1)
                new_beams.append(child)
                for w in range(beam_width):
                    pid = beam_id[b, w].item()
                    new_gen[b][w] = gen_tokens[b][pid] + [token_id[b, w].item()]
            beams = torch.stack(new_beams, dim=0)
            gen_tokens = new_gen

    generated = beams[:, :, -N_LAYERS:]
    return generated, beam_scores


@torch.no_grad()
def generate_recommendations(model, input_ids, beam_width, sid2item, device, trie=None):
    """
    beam search + invalid过滤，返回每个用户的推荐item列表。
    return:
      rec_lists: list[list[int]] 每个用户的有效item推荐(按beam分数排序)
      invalid_count, total_count: 用于算invalid rate
    """
    generated, scores = beam_search(model, input_ids, beam_width, device, trie=trie)
    B = generated.size(0)

    rec_lists = []
    invalid_count, total_count = 0, 0
    for b in range(B):
        recs = []
        seen_items = set()
        for w in range(beam_width):
            tokens = generated[b, w].tolist()          # 4个带偏移的token
            sid = tuple(tokens_to_sid(tokens))         # 还原成(k1,k2,k3,k4)
            total_count += 1
            # 1. 查表: 这个SID对应真实item吗?
            if sid in sid2item:
                item = sid2item[sid]
                # 2. 去重: 同一item可能被多个beam生成，只留一次
                if item not in seen_items:
                    recs.append(item)
                    seen_items.add(item)
            else:
                invalid_count += 1                     # 生成了不存在的SID
        rec_lists.append(recs)
    return rec_lists, invalid_count, total_count