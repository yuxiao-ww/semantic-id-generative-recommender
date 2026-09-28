"""
SID 前缀树: 把全部item的SID组织成Trie，供beam search做约束解码。

作用: 生成第l层token时，只允许选择"当前前缀在Trie中存在的后继token"，
      保证生成的4个token一定对应真实item(warm或cold)，且invalid rate=0。

token是带层偏移的 (第l层token = 码字 + l*256)，Trie直接用带偏移的token做key。
"""

import json
from gr.dataset import sid_to_tokens


class SidTrie:
    def __init__(self, item_sid_path):
        """
        item_sid_path: item_sid.json 或 item_sid_cold.json
        构建: 每个item的4位SID -> 带偏移token序列 -> 插入Trie
        """
        # 1. Trie用嵌套dict实现: {token: {child_token: {...}}}
        #    叶子节点存 item_idx
        self.root = {}
        with open(item_sid_path) as f:
            item_sid = json.load(f)

        for item_idx, sid in item_sid.items():
            tokens = sid_to_tokens(sid)          # 4个带偏移token
            node = self.root
            for t in tokens:
                if t not in node:
                    node[t] = {}
                node = node[t]
            # 2. 叶子: 记录这条路径对应的item (可能多个item共享前3位，但第4位去重后唯一)
            node["__item__"] = int(item_idx)

    def get_valid_next(self, prefix_tokens):
        """
        给定已生成的前缀token序列，返回合法的下一个token集合。
        prefix_tokens: list[int]，已生成的带偏移token (可能为空=第1层)
        return: set[int] 合法后继token，如果前缀非法则返回空set
        """
        node = self.root
        # 1. 沿前缀走到对应节点
        for t in prefix_tokens:
            if t not in node:
                return set()                     # 前缀不在Trie中，无合法后继
            node = node[t]
        # 2. 当前节点的所有孩子token(排除__item__标记)就是合法后继
        return set(k for k in node.keys() if k != "__item__")

    def get_item(self, tokens):
        """4个token走到叶子，返回对应item_idx，非法则None。"""
        node = self.root
        for t in tokens:
            if t not in node:
                return None
            node = node[t]
        return node.get("__item__", None)