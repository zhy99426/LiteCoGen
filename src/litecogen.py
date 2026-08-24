"""LiteCoGen: lightweight collaborative posterior correction for generative recommendation."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tiger import TigerRec
from .utils import as_batch, _history_to_item0


class TigerLiteCoGen(TigerRec):

    def __init__(
        self,
        *args: Any,
        be_energy_weight: float = 1.0,
        codebook_embedding_dim: int = 64,
        posterior_temperature: float = 1.0,
        posterior_step_size: float = 0.5,
        temporal_gamma: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.be_energy_weight = float(be_energy_weight)
        self.posterior_temperature = float(posterior_temperature)
        self.posterior_step_size = float(posterior_step_size)
        self.temporal_gamma = float(temporal_gamma)

        d = max(1, int(codebook_embedding_dim))
        L = int(self.item_codes.shape[1])
        C = int(self.codebook_size)

        self._A_u = nn.Parameter(torch.empty(L, C, d))
        self._A_v = nn.Parameter(torch.empty(L, C, d))
        self._a = nn.Parameter(torch.zeros(L, C))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self._A_u)
        nn.init.xavier_uniform_(self._A_v)
        nn.init.zeros_(self._a)

    def _item_repr(self, item_idx_0: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        L, C, d = codebook.shape
        tokens = self.item_codes[item_idx_0.clamp(min=0, max=self.n_items - 1)]
        flat_tokens = tokens + torch.arange(0, L * C, C, device=tokens.device)
        gathered = codebook.reshape(L * C, d)[flat_tokens]
        return gathered.sum(dim=-2) / (L ** 0.5)

    def _semid_bias(self, item_idx_0: torch.Tensor) -> torch.Tensor:
        L, C = self._a.shape
        tokens = self.item_codes[item_idx_0.clamp(min=0, max=self.n_items - 1)]
        flat_tokens = tokens + torch.arange(0, L * C, C, device=tokens.device)
        return self._a.reshape(L * C)[flat_tokens].sum(dim=-1)

    def _embed_context(self, item_ids_0idx: torch.Tensor) -> torch.Tensor:
        return self._item_repr(item_ids_0idx, self._A_u)

    def _embed_candidate(self, item_ids_0idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self._item_repr(item_ids_0idx, self._A_v)
        bias = self._semid_bias(item_ids_0idx)
        return emb, bias

    def _source_prior(self, T: int) -> torch.Tensor:
        pos = torch.arange(T, device=self._A_u.device).float()
        a = self.temporal_gamma ** (T - 1 - pos)
        return a / a.sum()

    def _history_encoding(self, il: torch.Tensor) -> torch.Tensor:
        m = il > 0
        H = il.shape[1]
        u = self._embed_context((il - 1).clamp(min=0))
        al = self._source_prior(H)
        w = al.unsqueeze(0) * m.float()
        return (u * w.unsqueeze(-1)).sum(dim=1) / w.sum(dim=1, keepdim=True).clamp(min=1e-8)

    def _extract_transition_pairs(
        self, history_1idx: torch.Tensor, target_1idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prev_list: list[int] = []
        next_list: list[int] = []
        for b in range(history_1idx.shape[0]):
            seq = [int(x) for x in history_1idx[b].cpu().tolist() if int(x) > 0]
            seq.append(int(target_1idx[b].item()))
            for t in range(len(seq) - 1):
                if seq[t] > 0 and seq[t + 1] > 0:
                    prev_list.append(seq[t] - 1)
                    next_list.append(seq[t + 1] - 1)
        if len(prev_list) < 2:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        return (
            torch.tensor(prev_list, dtype=torch.long),
            torch.tensor(next_list, dtype=torch.long),
        )

    def _behavioral_energy_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        rec_batch = as_batch(batch)
        device = rec_batch.item_id_list.device

        prev_0idx, next_0idx = self._extract_transition_pairs(
            rec_batch.item_id_list, rec_batch.item_id
        )
        P = prev_0idx.shape[0]
        if P < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        prev_t = prev_0idx.to(device)
        next_t = next_0idx.to(device)

        q = self._embed_context(prev_t)
        e_beh, bias = self._embed_candidate(next_t)

        S = q @ e_beh.T + bias.unsqueeze(0)
        labels = torch.arange(P, device=device)
        return F.cross_entropy(S, labels)

    def calculate_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        loss_ce = TigerRec.calculate_loss(self, batch)
        if self.be_energy_weight > 0:
            loss_be = self._behavioral_energy_loss(batch)
            return loss_ce + self.be_energy_weight * loss_be
        return loss_ce

    def calculate_val_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return TigerRec.calculate_loss(self, batch)

    def _rerank_row(
        self,
        history_row: torch.Tensor,
        pred_tokens_row: torch.Tensor,
        beam_scores_row: torch.Tensor,
        max_k: int,
    ) -> list[int]:
        device = beam_scores_row.device

        candidates: dict[int, float] = {}
        for beam_idx in range(pred_tokens_row.shape[0]):
            seq = tuple(
                int(x) for x in pred_tokens_row[beam_idx].detach().cpu().tolist()
            )
            item_id = self._token_to_item.get(seq)
            if item_id is None:
                continue
            score = float(beam_scores_row[beam_idx].item())
            old = candidates.get(int(item_id))
            if old is None or score > old:
                candidates[int(item_id)] = score

        if not candidates:
            return [0] * int(max_k)

        item_ids = list(candidates.keys())
        beam_vec = torch.tensor(
            [candidates[i] for i in item_ids], dtype=torch.float32, device=device
        )

        reward = torch.zeros_like(beam_vec)
        hist_0idx = _history_to_item0(history_row)
        if len(hist_0idx) > 0:
            hist_t = torch.tensor(hist_0idx, dtype=torch.long, device=device)
            cand_t = torch.tensor(item_ids, dtype=torch.long, device=device)

            u_hist = self._embed_context(hist_t)
            v_cand, bias_cand = self._embed_candidate(cand_t)
            M = u_hist @ v_cand.T + bias_cand.unsqueeze(0)

            P = F.softmax(M, dim=-1)

            alpha = self._source_prior(len(hist_0idx))

            p_trans = (alpha.unsqueeze(-1) * P).sum(dim=0)
            reward = torch.log(p_trans + 1e-8)

        tau = max(float(self.posterior_temperature), 1e-6)
        eta = float(self.posterior_step_size)
        logp = F.log_softmax(beam_vec / tau, dim=0)
        final_scores = logp + eta * reward

        order = torch.argsort(final_scores, descending=True).detach().cpu().tolist()
        ranked = [int(item_ids[idx]) for idx in order[:max_k]]
        ranked.extend(
            [ranked[-1] if ranked else 0] * max(0, int(max_k) - len(ranked))
        )
        return ranked

    def _reranked_items(
        self, batch: dict[str, torch.Tensor], beam_size: int, max_k: int
    ) -> torch.Tensor:
        pred_tokens, seq_scores = self._generate_sequences(batch, beam_size=beam_size)
        rec_batch = as_batch(batch)
        rows: list[list[int]] = []
        for row_idx in range(pred_tokens.shape[0]):
            rows.append(
                self._rerank_row(
                    history_row=rec_batch.item_id_list[row_idx],
                    pred_tokens_row=pred_tokens[row_idx],
                    beam_scores_row=seq_scores[row_idx],
                    max_k=max_k,
                )
            )
        return torch.tensor(rows, dtype=torch.long, device=seq_scores.device)

    def full_sort_predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        beam_size = max(self.trie_beam_width, 20)
        ranked = self._reranked_items(batch, beam_size=beam_size, max_k=beam_size)
        scores = torch.full(
            (ranked.shape[0], self.n_items), -1e9, dtype=torch.float32, device=ranked.device,
        )
        base_score = float(beam_size)
        for row_idx in range(ranked.shape[0]):
            seen: set[int] = set()
            rank_pos = 0
            for item_id in ranked[row_idx].tolist():
                item = int(item_id)
                if item < 0 or item >= self.n_items or item in seen:
                    continue
                scores[row_idx, item] = base_score - float(rank_pos)
                rank_pos += 1
                seen.add(item)
        empty_rows = scores.eq(-1e9).all(dim=1)
        if bool(empty_rows.any()):
            scores[empty_rows] = 0.0
        return scores

    def generate_topk(self, batch: dict[str, torch.Tensor], k: int = 20) -> torch.Tensor:
        topk = min(max(1, int(k)), self.n_items)
        beam_size = max(self.trie_beam_width, topk)
        ranked = self._reranked_items(batch, beam_size=beam_size, max_k=topk)
        return ranked[:, :topk].clamp(min=0, max=max(0, self.n_items - 1))

    def get_optimizer_param_groups(
        self, base_lr: float, weight_decay: float
    ) -> list[dict[str, object]]:
        reward_prefixes = ("_A_u", "_A_v", "_a")
        reward_params: list[nn.Parameter] = []
        base_params: list[nn.Parameter] = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(reward_prefixes):
                reward_params.append(param)
            else:
                base_params.append(param)

        groups: list[dict[str, object]] = [
            {"params": base_params, "lr": base_lr, "weight_decay": weight_decay},
        ]
        if reward_params:
            groups.append({
                "params": reward_params,
                "lr": base_lr,
                "weight_decay": weight_decay,
            })
        return groups
