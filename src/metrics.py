"""Evaluation metrics: Recall@K, NDCG@K, MRR."""
from __future__ import annotations

from collections.abc import Sequence

import torch


def _mrr_at_k(hits: torch.Tensor, k: int) -> float:
    capped_k = max(1, min(int(k), int(hits.shape[1])))
    hit_pos = hits[:, :capped_k]
    any_hit = hit_pos.any(dim=1)
    reciprocal = torch.zeros(hits.shape[0], device=hits.device)
    if bool(any_hit.any()):
        ranks = hit_pos.float().argmax(dim=1).float() + 1.0
        reciprocal[any_hit] = 1.0 / ranks[any_hit]
    return float(reciprocal.mean().item())


def ranking_metrics_from_topk_indices(
    topk_indices: torch.Tensor,
    target: torch.Tensor,
    k_values: Sequence[int] = (1, 5, 10, 20),
) -> dict[str, float]:
    if topk_indices.dim() != 2:
        raise ValueError(f"topk_indices must be 2D, got shape {tuple(topk_indices.shape)}")

    target_col = target.long().to(topk_indices.device) - 1
    max_k = int(topk_indices.shape[1])
    hits = topk_indices.eq(target_col.view(-1, 1))

    result: dict[str, float] = {}
    for k in k_values:
        actual_k = min(int(k), max_k)
        hit_at_k = hits[:, :actual_k].any(dim=1).float()
        result[f"recall@{k}"] = float(hit_at_k.mean().item())
        rank_hits = hits[:, :actual_k].float()
        discounts = 1.0 / torch.log2(torch.arange(actual_k, device=topk_indices.device).float() + 2.0)
        dcg = (rank_hits * discounts).sum(dim=1)
        result[f"ndcg@{k}"] = float(dcg.mean().item())

    mrr = _mrr_at_k(hits, max_k)
    result["mrr"] = mrr
    for k in k_values:
        result[f"mrr@{k}"] = _mrr_at_k(hits, min(int(k), max_k))
    return result


def ranking_metrics(
    scores: torch.Tensor,
    target: torch.Tensor,
    k_values: Sequence[int] = (1, 5, 10, 20),
) -> dict[str, float]:
    target_col = target.long().to(scores.device) - 1
    max_k = max(1, max(int(k) for k in k_values))
    max_k = min(max_k, scores.shape[1])
    _, topk = torch.topk(scores, k=max_k, dim=1)
    hits = topk.eq(target_col.view(-1, 1))
    result: dict[str, float] = {}
    for k in k_values:
        actual_k = min(int(k), scores.shape[1])
        hit_at_k = hits[:, :actual_k].any(dim=1).float()
        result[f"recall@{k}"] = float(hit_at_k.mean().item())
        rank_hits = hits[:, :actual_k].float()
        discounts = 1.0 / torch.log2(torch.arange(actual_k, device=scores.device).float() + 2.0)
        dcg = (rank_hits * discounts).sum(dim=1)
        result[f"ndcg@{k}"] = float(dcg.mean().item())
    mrr = _mrr_at_k(hits, max_k)
    result["mrr"] = mrr
    return result
