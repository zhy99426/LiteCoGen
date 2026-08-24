"""Sequential recommendation dataset."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch.utils.data import Dataset


def _right_pad(values: Iterable[int], max_len: int) -> tuple[list[int], int]:
    seq = list(values)[-int(max_len):]
    length = len(seq)
    if length < int(max_len):
        seq.extend([0] * (int(max_len) - length))
    return seq, length


def load_interactions(data_root: str | Path, dataset: str) -> dict[int, list[int]]:
    path = Path(data_root) / dataset / f"{dataset}.inter.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing interaction file: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(uid): [int(item) for item in items] for uid, items in raw.items()}


def load_items(data_root: str | Path, dataset: str) -> dict[int, dict]:
    path = Path(data_root) / dataset / f"{dataset}.item.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing item file: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(item): value for item, value in raw.items()}


def item_count(data_root: str | Path, dataset: str, inters: Mapping[int, list[int]]) -> int:
    inter_max = max((max(items) for items in inters.values() if items), default=-1)
    try:
        item_max = max(load_items(data_root, dataset).keys(), default=-1)
    except FileNotFoundError:
        item_max = -1
    return max(inter_max, item_max) + 1


class SeqRecDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        dataset: str,
        max_his_len: int = 20,
        mode: str = "train",
        seed: int = 42,
    ) -> None:
        if mode not in {"train", "valid", "test"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.data_root = Path(data_root)
        self.dataset = dataset
        self.max_his_len = int(max_his_len)
        self.mode = mode
        self.seed = int(seed)
        self.epoch = 0
        self.inters = load_interactions(self.data_root, self.dataset)
        self.user_num = max(self.inters.keys(), default=-1) + 1
        self.item_num = item_count(self.data_root, self.dataset, self.inters)
        self.samples = self._build_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return dict(self.samples[index])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_samples(self) -> list[dict[str, torch.Tensor]]:
        samples: list[dict[str, torch.Tensor]] = []
        for uid in sorted(self.inters):
            items = [item + 1 for item in self.inters[uid]]
            if len(items) < 3:
                continue
            if self.mode == "train":
                train_items = items[:-2]
                for end in range(1, len(train_items)):
                    padded, length = _right_pad(train_items[:end], self.max_his_len)
                    samples.append({
                        "user_id": torch.tensor(uid, dtype=torch.long),
                        "item_id_list": torch.tensor(padded, dtype=torch.long),
                        "item_length": torch.tensor(length, dtype=torch.long),
                        "item_id": torch.tensor(train_items[end], dtype=torch.long),
                        "history_items": torch.tensor(padded, dtype=torch.long),
                    })
            elif self.mode == "valid":
                padded, length = _right_pad(items[:-2], self.max_his_len)
                samples.append({
                    "user_id": torch.tensor(uid, dtype=torch.long),
                    "item_id_list": torch.tensor(padded, dtype=torch.long),
                    "item_length": torch.tensor(length, dtype=torch.long),
                    "item_id": torch.tensor(items[-2], dtype=torch.long),
                    "history_items": torch.tensor(padded, dtype=torch.long),
                })
            else:
                padded, length = _right_pad(items[:-1], self.max_his_len)
                samples.append({
                    "user_id": torch.tensor(uid, dtype=torch.long),
                    "item_id_list": torch.tensor(padded, dtype=torch.long),
                    "item_length": torch.tensor(length, dtype=torch.long),
                    "item_id": torch.tensor(items[-1], dtype=torch.long),
                    "history_items": torch.tensor(padded, dtype=torch.long),
                })
        return samples
