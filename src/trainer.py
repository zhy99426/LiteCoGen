"""Training loop and semantic ID construction."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .metrics import ranking_metrics_from_topk_indices
from .dataset import SeqRecDataset
from .rqvae import RQVAE
from .utils import as_batch


def build_dataloaders(
    data_root: str,
    dataset: str,
    max_his_len: int = 20,
    batch_size: int = 256,
    eval_batch_size: int = 256,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, int, int]:
    train_set = SeqRecDataset(data_root, dataset, max_his_len=max_his_len, mode="train", seed=seed)
    valid_set = SeqRecDataset(data_root, dataset, max_his_len=max_his_len, mode="valid", seed=seed)
    test_set = SeqRecDataset(data_root, dataset, max_his_len=max_his_len, mode="test", seed=seed)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0)
    valid_loader = DataLoader(valid_set, batch_size=eval_batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=eval_batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, valid_loader, test_loader, train_set.user_num, train_set.item_num


def build_semantic_ids(
    data_root: str,
    dataset: str,
    n_items: int,
    embedding_size: int = 64,
    num_codebooks: int = 3,
    codebook_size: int = 256,
    semid_mode: str = "rqvae",
    semid_seed: int = 42,
    semid_kmeans_iters: int = 20,
    semid_rqvae_epochs: int = 50,
    semid_rqvae_batch_size: int = 1024,
    semid_rqvae_lr: float = 1e-3,
    semid_rqvae_weight_decay: float = 1e-5,
    semid_rqvae_hidden_size: int = 256,
    semid_rqvae_code_dim: int = 64,
    semid_rqvae_commitment_weight: float = 0.25,
    semid_rqvae_quant_loss_weight: float = 1.0,
    semid_rqvae_device: str = "cuda",
    cache_dir: str | None = None,
) -> tuple[torch.Tensor, RQVAE | None]:
    import numpy as np

    cache_root = Path(cache_dir) if cache_dir else Path(data_root) / dataset
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{dataset}.semid-{semid_mode}-k{codebook_size}-m{num_codebooks}.npy"
    rqvae_path = cache_root / f"{dataset}.semid-{semid_mode}-k{codebook_size}-m{num_codebooks}.rqvae.pt"

    if cache_path.exists():
        codes = torch.from_numpy(np.load(cache_path)).long()
        if codes.shape[0] == n_items and semid_mode != "rqvae":
            return codes, None
        if codes.shape[0] == n_items and semid_mode == "rqvae" and rqvae_path.exists():
            pass

    emb_path = Path(data_root) / dataset / f"{dataset}.emb-llama-td.npy"
    if emb_path.exists():
        embeddings = torch.from_numpy(np.load(emb_path)).float()
        if embeddings.shape[0] < n_items:
            pad = torch.zeros(n_items - embeddings.shape[0], embeddings.shape[1])
            embeddings = torch.cat([embeddings, pad], dim=0)
        embeddings = embeddings[:n_items]
    else:
        generator = torch.Generator().manual_seed(42)
        embeddings = torch.randn(n_items, embedding_size, generator=generator)

    if embeddings.shape[1] > embedding_size:
        embeddings = embeddings[:, :embedding_size]
    elif embeddings.shape[1] < embedding_size:
        pad = torch.zeros(embeddings.shape[0], embedding_size - embeddings.shape[1])
        embeddings = torch.cat([embeddings, pad], dim=1)

    device = torch.device(semid_rqvae_device if semid_rqvae_device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = RQVAE(
        in_dim=int(embeddings.shape[1]),
        codebook_size_list=[codebook_size] * num_codebooks,
        codebook_dim=max(8, min(semid_rqvae_code_dim, embeddings.shape[1])),
        quant_loss_weight=semid_rqvae_quant_loss_weight,
        commitment_weight=semid_rqvae_commitment_weight,
        kmeans_iters=max(5, semid_kmeans_iters),
        base_embeddings=embeddings,
        hidden_size=max(16, semid_rqvae_hidden_size),
    ).to(device)

    model.kmeans_init(seed=semid_seed)

    item_ids = torch.arange(1, embeddings.shape[0] + 1, dtype=torch.long)
    dataset_t = torch.utils.data.TensorDataset(item_ids)
    loader = DataLoader(dataset_t, batch_size=semid_rqvae_batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(semid_seed), drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=semid_rqvae_lr, weight_decay=semid_rqvae_weight_decay)

    total_epochs = max(1, semid_rqvae_epochs)
    best_codes = None
    best_state = None
    best_collision = float("inf")
    started = time.perf_counter()

    for epoch in range(total_epochs):
        model.train()
        for (batch_ids,) in loader:
            batch_ids = batch_ids.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, _, total_loss, _, _, _ = model(batch_ids)
            if not torch.isfinite(total_loss).item():
                continue
            total_loss.backward()
            optimizer.step()

        model.eval()
        model.build_indices()
        _, collision_rate = model.get_collision_rate()
        current_codes = model.indices_tensor(include_unique_suffix=True, include_padding=False).cpu()

        if float(collision_rate) < best_collision or best_codes is None:
            best_collision = float(collision_rate)
            best_codes = current_codes
            best_state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}

        if (epoch + 1) % max(1, total_epochs // 10) == 0 or epoch == 0:
            elapsed = time.perf_counter() - started
            print(f"  [RQ-VAE] epoch {epoch + 1}/{total_epochs} loss={float(total_loss.item()):.4f} "
                  f"collision={float(collision_rate):.4f} best_col={best_collision:.4f} elapsed={elapsed:.1f}s")

    assert best_codes is not None and best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    model.build_indices()

    np.save(cache_path, best_codes.numpy())
    torch.save({"state_dict": best_state, "n_items": n_items, "codebook_size": codebook_size, "num_codebooks": num_codebooks}, rqvae_path)

    return best_codes, model


def load_or_build_semids(
    data_root: str,
    dataset: str,
    n_items: int,
    embedding_size: int = 64,
    num_codebooks: int = 3,
    codebook_size: int = 256,
    semid_mode: str = "rqvae",
    **kwargs,
) -> tuple[torch.Tensor, RQVAE | None]:
    import numpy as np, json
    cache_root = Path(data_root) / dataset
    cache_path = cache_root / f"{dataset}.semid-{semid_mode}-k{codebook_size}-m{num_codebooks}.npy"
    rqvae_path = cache_root / f"{dataset}.semid-{semid_mode}-k{codebook_size}-m{num_codebooks}.rqvae.pt"
    meta_path = cache_root / f"{dataset}.semid-{semid_mode}-k{codebook_size}-m{num_codebooks}.meta.json"

    if cache_path.exists():
        codes = torch.from_numpy(np.load(cache_path)).long()
        if codes.shape[0] == n_items:
            if semid_mode == "rqvae" and rqvae_path.exists():
                meta = {}
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                actual_embed_dim = int(meta.get("embedding_used_dim", meta.get("embedding_raw_dim", embedding_size)))
                actual_code_dim = int(meta.get("rqvae_code_dim", 32))
                actual_hidden_size = int(meta.get("rqvae_hidden_size", 256))
                actual_kmeans_iters = int(meta.get("rqvae_kmeans_iters", 20))

                emb_path = Path(data_root) / dataset / f"{dataset}.emb-llama-td.npy"
                if emb_path.exists():
                    embeddings = torch.from_numpy(np.load(emb_path)).float()
                    if embeddings.shape[0] < n_items:
                        pad = torch.zeros(n_items - embeddings.shape[0], embeddings.shape[1])
                        embeddings = torch.cat([embeddings, pad], dim=0)
                    embeddings = embeddings[:n_items]
                    if embeddings.shape[1] != actual_embed_dim:
                        if embeddings.shape[1] > actual_embed_dim:
                            embeddings = embeddings[:, :actual_embed_dim]
                        else:
                            pad = torch.zeros(embeddings.shape[0], actual_embed_dim - embeddings.shape[1])
                            embeddings = torch.cat([embeddings, pad], dim=1)
                else:
                    embeddings = torch.randn(n_items, actual_embed_dim)

                payload = torch.load(rqvae_path, map_location="cpu")
                rqvae = RQVAE(
                    in_dim=actual_embed_dim,
                    codebook_size_list=[codebook_size] * num_codebooks,
                    codebook_dim=actual_code_dim,
                    base_embeddings=embeddings,
                    hidden_size=actual_hidden_size,
                    kmeans_iters=actual_kmeans_iters,
                )
                rqvae.load_state_dict(payload["state_dict"])
                rqvae.eval()
                rqvae.build_indices()
                return codes, rqvae
            return codes, None

    semid_kwargs = {}
    key_map = {
        "kmeans_iters": "semid_kmeans_iters",
        "rqvae_epochs": "semid_rqvae_epochs",
        "rqvae_batch_size": "semid_rqvae_batch_size",
        "rqvae_lr": "semid_rqvae_lr",
        "rqvae_weight_decay": "semid_rqvae_weight_decay",
        "rqvae_hidden_size": "semid_rqvae_hidden_size",
        "rqvae_code_dim": "semid_rqvae_code_dim",
        "rqvae_commitment_weight": "semid_rqvae_commitment_weight",
        "rqvae_quant_loss_weight": "semid_rqvae_quant_loss_weight",
        "rqvae_device": "semid_rqvae_device",
    }
    for k, v in kwargs.items():
        semid_kwargs[key_map.get(k, k)] = v
    return build_semantic_ids(
        data_root=data_root, dataset=dataset, n_items=n_items,
        embedding_size=embedding_size, num_codebooks=num_codebooks,
        codebook_size=codebook_size, semid_mode=semid_mode, **semid_kwargs,
    )


def train(
    model: TigerLiteCoGen,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    output_dir: str,
    max_epochs: int = 200,
    patience: int = 5,
    lr: float = 5e-4,
    weight_decay: float = 0.0,
    device: str = "cuda",
    seed: int = 42,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if hasattr(model, 'get_optimizer_param_groups'):
        param_groups = model.get_optimizer_param_groups(lr, weight_decay)
    else:
        param_groups = [{"params": model.parameters(), "lr": lr, "weight_decay": weight_decay}]
    optimizer = torch.optim.AdamW(param_groups)

    best_val_loss = float("inf")
    best_epoch = 0
    best_ckpt_path = output_path / "best.pt"
    patience_counter = 0

    model = model.to(device)

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loader.dataset.set_epoch(epoch)
        total_loss = 0.0
        start = time.perf_counter()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model.calculate_loss(batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                val_loss += float(model.calculate_val_loss(batch).item())

        avg_train = total_loss / max(1, len(train_loader))
        avg_val = val_loss / max(1, len(valid_loader))
        elapsed = time.perf_counter() - start

        print(f"Epoch {epoch:3d}/{max_epochs} | train_loss={avg_train:.4f} val_loss={avg_val:.4f} | {elapsed:.1f}s")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            patience_counter = 0
            model.save(str(best_ckpt_path))
            print(f"  -> best checkpoint saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Loading best checkpoint (epoch {best_epoch})")
    model.load(str(best_ckpt_path), map_location=device)
    model = model.to(device)
    model.eval()

    all_topk = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            topk = model.generate_topk(batch, k=20)
            all_topk.append(topk.cpu())
            all_targets.append(as_batch(batch).item_id.cpu())

    topk_tensor = torch.cat(all_topk, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)
    metrics = ranking_metrics_from_topk_indices(topk_tensor, targets_tensor, k_values=(1, 5, 10))

    print("\n=== Test Results ===")
    for k in [1, 5, 10]:
        print(f"  NDCG@{k:2d} = {metrics.get(f'ndcg@{k}', 0):.4f}")
    for k in [5, 10]:
        print(f"  HR@{k:2d}   = {metrics.get(f'recall@{k}', 0):.4f}")

    result = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "ndcg@1": metrics.get("ndcg@1", 0),
        "ndcg@5": metrics.get("ndcg@5", 0),
        "ndcg@10": metrics.get("ndcg@10", 0),
        "hr@5": metrics.get("recall@5", 0),
        "hr@10": metrics.get("recall@10", 0),
    }
    with open(output_path / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result
