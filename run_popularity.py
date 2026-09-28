"""
用 Popularity 跑通评测器。这一步真正的目的是验证 S2b 写对了。
"""

import os
import torch
from torch.utils.data import DataLoader

from data.dataset import SeqRecDataset, eval_collate
from baselines.popularity import PopularityModel
from eval.evaluator import evaluate, print_metrics

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    device = get_device()
    data_dir = f"{PROJECT_ROOT}/data/processed/beauty"

    # 1. train split 只用来 fit 频次；test split 用来评测
    train_ds = SeqRecDataset(data_dir, split="train", max_len=50)
    test_ds = SeqRecDataset(data_dir, split="test", max_len=50)
    test_loader = DataLoader(test_ds, batch_size=256, collate_fn=eval_collate)

    # 2. fit
    model = PopularityModel(num_items=train_ds.num_items, device=device)
    model.fit(train_ds)

    # 3. 评测
    result = evaluate(model, test_loader, num_items=train_ds.num_items,
                      device=device, ks=(5, 10, 20))
    print_metrics(result, tag="Popularity")


if __name__ == "__main__":
    main()