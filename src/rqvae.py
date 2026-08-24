"""RQ-VAE for semantic ID construction."""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinkhorn_algorithm(distances: torch.Tensor, epsilon: float = 0.05, sinkhorn_iterations: int = 50) -> torch.Tensor:
    q = torch.exp(-distances / epsilon)
    q = q / (q.sum() + 1e-8)
    b, k = q.shape
    for _ in range(sinkhorn_iterations):
        q = q / (q.sum(dim=1, keepdim=True) + 1e-8)
        q = q / b
        q = q / (q.sum(dim=0, keepdim=True) + 1e-8)
        q = q / k
    return q * b


class MLPLayers(nn.Module):
    def __init__(self, layers, dropout: float = 0.0, activation: str = "relu", bn: bool = False):
        super().__init__()
        self.mlp_layers = self._build_mlp(layers, dropout, activation, bn)
        self.apply(self._init_weights)

    def _build_mlp(self, layers, dropout: float, activation: str, bn: bool):
        act_map = {"relu": nn.ReLU(), "leakyrelu": nn.LeakyReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh()}
        act = act_map.get(activation.lower() if isinstance(activation, str) else "", None)
        modules = []
        for i in range(len(layers) - 1):
            in_dim, out_dim = layers[i], layers[i + 1]
            modules.append(nn.Linear(in_dim, out_dim))
            if bn:
                modules.append(nn.BatchNorm1d(out_dim))
            if act is not None and i != len(layers) - 2:
                modules.append(act)
            if dropout > 0.0:
                modules.append(nn.Dropout(dropout))
        return nn.Sequential(*modules)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.mlp_layers(x)


def _run_kmeans(data: torch.Tensor, n_clusters: int, n_iters: int, seed: int = 42) -> torch.Tensor:
    values = data.detach().cpu().numpy()
    n_clusters = max(1, min(int(n_clusters), int(values.shape[0])))
    try:
        from sklearn.cluster import KMeans
        model = KMeans(n_clusters=n_clusters, max_iter=max(1, int(n_iters)), n_init=10, random_state=int(seed))
        model.fit(values)
        return torch.from_numpy(model.cluster_centers_).to(data.device, dtype=data.dtype)
    except Exception:
        generator = torch.Generator(device=data.device).manual_seed(int(seed))
        if data.shape[0] <= n_clusters:
            pick = torch.arange(data.shape[0], device=data.device)
            if data.shape[0] < n_clusters:
                extra = torch.randint(data.shape[0], (n_clusters - data.shape[0],), generator=generator, device=data.device)
                pick = torch.cat([pick, extra], dim=0)
            return data[pick]
        init_idx = torch.randperm(data.shape[0], generator=generator, device=data.device)[:n_clusters]
        centers = data[init_idx].clone()
        for _ in range(max(1, int(n_iters))):
            dist = torch.cdist(data, centers)
            assign = dist.argmin(dim=-1)
            updated = centers.clone()
            for cid in range(n_clusters):
                mask = assign.eq(cid)
                if bool(mask.any()):
                    updated[cid] = data[mask].mean(dim=0)
            centers = updated
        return centers


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int = 256, codebook_dim: int = 32,
                 mu: float = 0.25, sk_epsilon: float = 0.0, sk_iters: int = 50) -> None:
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.codebook_dim = int(codebook_dim)
        self.mu = float(mu)
        self.sk_epsilon = float(sk_epsilon)
        self.sk_iters = int(sk_iters)
        self.embedding = nn.Embedding(self.codebook_size, self.codebook_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.embedding.weight)

    def _compute_distances(self, x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        return x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=-1, keepdim=True).t() - 2.0 * x @ codebook.t()

    def _normalize_distances(self, d2: torch.Tensor) -> torch.Tensor:
        max_d, min_d = d2.max(), d2.min()
        mid = (max_d + min_d) / 2.0
        return (d2 - mid) / (max_d - min_d + 1e-6)

    def _assign_code(self, d2: torch.Tensor, *, use_sinkhorn: bool | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if use_sinkhorn is None:
            use_sinkhorn = self.sk_epsilon > 0.0
        if bool(use_sinkhorn) and self.sk_epsilon > 0.0:
            d2_normed = self._normalize_distances(d2).double()
            probs = sinkhorn_algorithm(d2_normed, self.sk_epsilon, self.sk_iters)
            return probs.to(dtype=d2.dtype), probs.argmax(dim=-1)
        probs = F.softmax(-d2, dim=-1)
        return probs, (-d2).argmax(dim=-1)

    def assign(self, x: torch.Tensor, *, use_sinkhorn: bool | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        is_seq = x.dim() == 3
        x_flat = x.view(-1, x.shape[-1]) if is_seq else x
        d2 = self._compute_distances(x_flat, self.embedding.weight)
        probs, indices = self._assign_code(d2, use_sinkhorn=use_sinkhorn)
        quant = self.embedding(indices)
        if is_seq:
            bsz, seq_len = x.shape[0], x.shape[1]
            indices = indices.view(bsz, seq_len)
            probs = probs.view(bsz, seq_len, self.codebook_size)
            quant = quant.view(bsz, seq_len, self.codebook_dim)
        return indices, probs, quant

    def forward(self, x: torch.Tensor, *, use_sinkhorn: bool | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices, probs, hard_quant = self.assign(x, use_sinkhorn=use_sinkhorn)
        is_seq = x.dim() == 3
        x_flat = x.view(-1, x.shape[-1]) if is_seq else x
        hard_quant_flat = hard_quant.view(-1, hard_quant.shape[-1]) if is_seq else hard_quant
        commitment = F.mse_loss(hard_quant_flat.detach(), x_flat, reduction="none").mean(dim=-1)
        codebook = F.mse_loss(hard_quant_flat, x_flat.detach(), reduction="none").mean(dim=-1)
        loss = codebook + self.mu * commitment
        quant_flat = (hard_quant_flat - x_flat).detach() + x_flat
        if is_seq:
            quant = quant_flat.view(x.shape[0], x.shape[1], self.codebook_dim)
            loss = loss.view(x.shape[0], x.shape[1])
        else:
            quant = quant_flat
        return indices, probs, quant, loss

    def kmeans_init(self, data: torch.Tensor, kmeans_iters: int = 20, seed: int = 42) -> torch.Tensor:
        centers = _run_kmeans(data, self.codebook_size, n_iters=kmeans_iters, seed=seed)
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.embedding.weight[:centers.shape[0]].copy_(centers)
            if centers.shape[0] < self.codebook_size:
                fill_size = self.codebook_size - centers.shape[0]
                idx = torch.arange(fill_size, device=data.device) % max(1, data.shape[0])
                self.embedding.weight[centers.shape[0]:].copy_(data[idx])
        d2 = self._compute_distances(data, self.embedding.weight)
        return self.embedding(d2.argmin(dim=-1))

    def get_codebook(self) -> torch.Tensor:
        return self.embedding.weight


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, codebook_size_list: list[int], codebook_dim: int,
                 *, mu: float = 0.25, sk_epsilons: list[float] | None = None, sk_iters: int = 50) -> None:
        super().__init__()
        self.codebook_size_list = [int(v) for v in codebook_size_list]
        self.codebook_dim = int(codebook_dim)
        self.sk_epsilons = sk_epsilons or [0.0] * len(self.codebook_size_list)
        if len(self.sk_epsilons) != len(self.codebook_size_list):
            raise ValueError("sk_epsilons length must equal codebook_size_list length.")
        self.vq_layers = nn.ModuleList([
            VectorQuantizer(codebook_size=size, codebook_dim=self.codebook_dim, mu=float(mu),
                            sk_epsilon=float(eps), sk_iters=int(sk_iters))
            for size, eps in zip(self.codebook_size_list, self.sk_epsilons)
        ])

    def forward(self, x: torch.Tensor, *, use_sinkhorn: bool | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = x
        quantized = torch.zeros_like(x)
        all_indices, all_probs, all_losses = [], [], []
        for quantizer in self.vq_layers:
            indices, probs, quants, losses = quantizer(residual, use_sinkhorn=use_sinkhorn)
            residual = residual - quants
            quantized = quantized + quants
            all_indices.append(indices); all_probs.append(probs); all_losses.append(losses)
        return torch.stack(all_indices, dim=-1), torch.stack(all_probs, dim=-1), quantized, torch.stack(all_losses, dim=-1)

    def encode_indices(self, x: torch.Tensor, *, use_sinkhorn: bool | None = None) -> torch.Tensor:
        residual = x
        all_indices = []
        for quantizer in self.vq_layers:
            indices, _, quants = quantizer.assign(residual, use_sinkhorn=use_sinkhorn)
            residual = residual - quants
            all_indices.append(indices)
        return torch.stack(all_indices, dim=-1)

    def kmeans_init(self, x: torch.Tensor, kmeans_iters: int = 20, seed: int = 42) -> None:
        residual = x
        for idx, quantizer in enumerate(self.vq_layers):
            quantized = quantizer.kmeans_init(residual, kmeans_iters=max(1, int(kmeans_iters)), seed=int(seed) + idx * 97)
            residual = residual - quantized

    def get_codebooks(self) -> torch.Tensor:
        return torch.stack([q.get_codebook() for q in self.vq_layers], dim=0)


class RQVAE(nn.Module):
    def __init__(self, *, in_dim: int, codebook_size_list: list[int], codebook_dim: int = 32,
                 layers: Sequence[int] | None = None, activation: str = "relu",
                 dropout_prob: float = 0.0, bn: bool = False,
                 quant_loss_weight: float = 1.0, commitment_weight: float = 0.25,
                 kmeans_iters: int = 20,
                 balance_weight: float = 0.0, entropy_weight: float = 0.0,
                 sk_epsilons: list[float] | None = None, sk_iters: int = 50,
                 base_embeddings: torch.Tensor | np.ndarray | None = None,
                 hidden_size: int = 256) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.codebook_size_list = [int(v) for v in codebook_size_list]
        self.codebook_dim = int(codebook_dim)
        self.quant_loss_weight = float(quant_loss_weight)
        self.kmeans_iters = int(kmeans_iters)
        self.balance_weight = float(balance_weight)
        self.entropy_weight = float(entropy_weight)
        self.sk_epsilons = sk_epsilons or [0.0] * len(self.codebook_size_list)
        self.sk_iters = int(sk_iters)
        if len(self.sk_epsilons) != len(self.codebook_size_list):
            raise ValueError("sk_epsilons length must equal codebook_size_list length.")
        if layers is None:
            layers = self._default_layers(self.in_dim, hidden_size=hidden_size, code_dim=self.codebook_dim)
        self.encode_layer_dims = [self.in_dim] + [int(v) for v in layers] + [self.codebook_dim]
        self.encoder = MLPLayers(self.encode_layer_dims, dropout_prob, activation, bn)
        self.rq = ResidualVectorQuantizer(self.codebook_size_list, self.codebook_dim,
                                          mu=float(commitment_weight), sk_epsilons=self.sk_epsilons, sk_iters=self.sk_iters)
        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(self.decode_layer_dims, dropout_prob, activation, bn)
        if base_embeddings is None:
            raise ValueError("base_embeddings is required for RQVAE.")
        base_tensor = torch.as_tensor(base_embeddings, dtype=torch.float32)
        if base_tensor.dim() != 2:
            raise ValueError("base_embeddings must be a 2D tensor [n_items, dim].")
        if base_tensor.shape[1] != self.in_dim:
            raise ValueError(f"base_embeddings dim mismatch: expected {self.in_dim}, got {base_tensor.shape[1]}")
        with_pad = torch.cat([torch.zeros(1, self.in_dim, dtype=base_tensor.dtype), base_tensor], dim=0)
        self.base_embedding = nn.Embedding(with_pad.shape[0], self.in_dim)
        self.base_embedding.weight.data.copy_(with_pad)
        self.base_embedding.weight.requires_grad = False
        self.all_indices: np.ndarray | None = None
        self.full_all_indices: list[list[int]] | None = None
        self.code2item: dict[str, list[int]] | None = None
        self.unique_indices: list[int] | None = None

    @staticmethod
    def _default_layers(in_dim: int, hidden_size: int, code_dim: int) -> list[int]:
        in_dim = int(in_dim); hidden_size = max(16, int(hidden_size)); code_dim = max(8, int(code_dim))
        if in_dim <= hidden_size:
            return [hidden_size]
        half = max(code_dim * 2, hidden_size // 2)
        return [hidden_size] if half == hidden_size else [hidden_size, half]

    def kmeans_init(self, seed: int = 42) -> None:
        with torch.no_grad():
            emb = self.base_embedding.weight.data[1:]
            self.rq.kmeans_init(self.encoder(emb), kmeans_iters=self.kmeans_iters, seed=seed)

    def forward(self, item_ids: torch.Tensor, *, use_sinkhorn: bool | None = None):
        mask = item_ids.ne(0)
        x_in = self.base_embedding(item_ids)
        encoded = self.encoder(x_in)
        hard_indices, soft_probs, x_q, quant_losses = self.rq(encoded, use_sinkhorn=use_sinkhorn)
        decoded = self.decoder(x_q)
        valid_quant_losses = quant_losses[mask]
        quant_loss = valid_quant_losses.mean() if valid_quant_losses.numel() > 0 else x_q.new_zeros(())
        diff = decoded - x_in
        recon_loss = diff[mask].pow(2).mean() if bool(mask.any()) else x_q.new_zeros(())
        total_loss = recon_loss + self.quant_loss_weight * quant_loss
        if self.balance_weight > 0 or self.entropy_weight > 0:
            probs = soft_probs[mask].float()
            if probs.numel() > 0:
                if probs.dim() != 3:
                    probs = probs.reshape(-1, probs.shape[-2], probs.shape[-1])
                probs_by_level = probs.permute(0, 2, 1).clamp_min(1e-8)
                usage = probs_by_level.mean(dim=0)
                cs = usage.shape[-1]; uniform = usage.new_full(usage.shape, 1.0 / max(1, cs))
                if self.balance_weight > 0:
                    total_loss = total_loss + self.balance_weight * ((usage - uniform).pow(2).mean() * cs)
                if self.entropy_weight > 0:
                    usage_entropy = -(usage * usage.log()).sum(dim=-1) / np.log(max(2, cs))
                    total_loss = total_loss + self.entropy_weight * (1.0 - usage_entropy).mean()
        return hard_indices, soft_probs, total_loss, recon_loss, quant_loss, quant_loss.new_zeros(())

    @torch.no_grad()
    def encode_items(self, item_ids: torch.Tensor, *, use_sinkhorn: bool | None = None) -> torch.Tensor:
        x_in = self.base_embedding(item_ids)
        return self.rq.encode_indices(self.encoder(x_in), use_sinkhorn=use_sinkhorn)

    @torch.no_grad()
    def build_indices(self) -> None:
        item_ids = torch.arange(self.base_embedding.weight.size(0), device=self.base_embedding.weight.device, dtype=torch.long)
        encoded_items = self.encoder(self.base_embedding(item_ids))
        hard_indices = self.rq.encode_indices(encoded_items)
        self.all_indices = hard_indices.detach().cpu().numpy()
        self.full_all_indices = [[-1] * (len(self.codebook_size_list) + 1)]
        self.code2item = defaultdict(list)
        self.unique_indices = [0]
        for i, code in enumerate(self.all_indices[1:]):
            row = [int(v) for v in code.tolist()]
            key = ",".join(map(str, row))
            self.code2item[key].append(i + 1)
            unique_index = len(self.code2item[key]) - 1
            self.full_all_indices.append(row + [int(unique_index)])
            self.unique_indices.append(int(unique_index))

    def get_collision_rate(self) -> tuple[int, float]:
        if self.code2item is None or self.all_indices is None:
            self.build_indices()
        total_items = len(self.all_indices) - 1
        unique_items = len(self.code2item)
        collision_rate = 1.0 - (unique_items / total_items) if total_items > 0 else 0.0
        return int(unique_items), float(collision_rate)

    def indices_tensor(self, *, include_unique_suffix: bool = True, include_padding: bool = False) -> torch.Tensor:
        if self.all_indices is None or self.full_all_indices is None:
            self.build_indices()
        rows = self.full_all_indices if include_unique_suffix else self.all_indices.tolist()
        tensor = torch.tensor(rows, dtype=torch.long)
        return tensor if include_padding else tensor[1:]
