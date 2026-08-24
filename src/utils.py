"""Utility functions: batch handling and helpers."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RecBatch:
    user_id: torch.Tensor
    item_id_list: torch.Tensor
    item_length: torch.Tensor
    item_id: torch.Tensor
    neg_item_id: torch.Tensor | None
    history_items: torch.Tensor


def as_batch(batch: dict[str, torch.Tensor]) -> RecBatch:
    return RecBatch(
        user_id=batch["user_id"].long(),
        item_id_list=batch["item_id_list"].long(),
        item_length=batch["item_length"].long(),
        item_id=batch["item_id"].long(),
        neg_item_id=batch.get("neg_item_id").long() if batch.get("neg_item_id") is not None else None,
        history_items=batch.get("history_items", batch["item_id_list"]).long(),
    )


def _history_to_item0(history_row: torch.Tensor) -> list[int]:
    return [int(x) - 1 for x in history_row.detach().cpu().tolist() if int(x) > 0]
