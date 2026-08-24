"""Training entry point."""
from __future__ import annotations

import argparse
import yaml
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.trainer import train, build_dataloaders, load_or_build_semids
from src.litecogen import TigerLiteCoGen
from src.tiger import TigerRec


def main():
    parser = argparse.ArgumentParser(description="LiteCoGen Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    parser.add_argument("--data_path", type=str, default="./data", help="Dataset root directory")
    parser.add_argument("--dataset", type=str, default="Toys_and_Games", help="Dataset name")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model", type=str, default="litecogen",
                        choices=["tiger", "litecogen"],
                        help="Model type")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["prepare", "semid", "train", "all"],
                        help="Pipeline stage")
    args = parser.parse_args()

    # Load config
    config = {}
    if Path(args.config).exists():
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}

    training_cfg = config.get("training", {})
    model_cfg = config.get("model_kwargs", {})
    semid_cfg = config.get("semid", {})

    data_path = args.data_path
    dataset = args.dataset
    device = args.device if torch.cuda.is_available() else "cpu"

    max_his_len = training_cfg.get("max_his_len", 20)
    batch_size = training_cfg.get("batch_size", 256)
    eval_batch_size = training_cfg.get("eval_batch_size", 256)
    embedding_size = training_cfg.get("embedding_size", 64)

    from src.dataset import item_count, load_interactions
    inters = load_interactions(data_path, dataset)
    n_items = item_count(data_path, dataset, inters)
    item_codes = None
    rqvae_tokenizer = None
    train_loader = valid_loader = test_loader = None

    if args.stage in ("prepare", "all"):
        print(f"Building dataloaders for {dataset}...")
        train_loader, valid_loader, test_loader, _n_users, n_items = build_dataloaders(
            data_root=data_path, dataset=dataset,
            max_his_len=max_his_len, batch_size=batch_size,
            eval_batch_size=eval_batch_size,
        )
        print(f"  Users: {_n_users}, Items: {n_items}")
        print(f"  Train batches: {len(train_loader)}, Valid: {len(valid_loader)}, Test: {len(test_loader)}")

    # Build semantic IDs
    num_codebooks = semid_cfg.get("num_codebooks", 3)
    codebook_size = semid_cfg.get("codebook_size", 256)
    semid_mode = semid_cfg.get("mode", "rqvae")

    if args.stage in ("semid", "all"):
        print(f"Building semantic IDs (mode={semid_mode}, codebooks={num_codebooks}, size={codebook_size})...")
        item_codes, rqvae_tokenizer = load_or_build_semids(
            data_root=data_path, dataset=dataset, n_items=n_items,
            embedding_size=embedding_size, num_codebooks=num_codebooks,
            codebook_size=codebook_size, semid_mode=semid_mode,
            semid_seed=args.seed,
            **{k: v for k, v in semid_cfg.items() if k not in ("mode", "num_codebooks", "codebook_size")},
        )
        print(f"  Item codes shape: {item_codes.shape}")
        if rqvae_tokenizer is not None:
            unique, col_rate = rqvae_tokenizer.get_collision_rate()
            print(f"  Unique codes: {unique}, collision rate: {col_rate:.4f}")

    # Load item_codes from cache if not already built (for standalone --stage train)
    if item_codes is None:
        item_codes, rqvae_tokenizer = load_or_build_semids(
            data_root=data_path, dataset=dataset, n_items=n_items,
            embedding_size=embedding_size, num_codebooks=num_codebooks,
            codebook_size=codebook_size, semid_mode=semid_mode,
            semid_seed=args.seed,
            **{k: v for k, v in semid_cfg.items() if k not in ("mode", "num_codebooks", "codebook_size")},
        )

    # Build dataloaders if not already done (for standalone --stage train)
    if train_loader is None:
        print(f"Building dataloaders for {dataset}...")
        train_loader, valid_loader, test_loader, _n_users, n_items = build_dataloaders(
            data_root=data_path, dataset=dataset,
            max_his_len=max_his_len, batch_size=batch_size,
            eval_batch_size=eval_batch_size,
        )
        print(f"  Users: {_n_users}, Items: {n_items}")

    if args.stage in ("train", "all"):
        model_type = args.model
        print(f"Initializing {model_type} model...")
        model_kwargs = {
            "n_items": n_items,
            "max_seq_length": max_his_len,
            "embedding_size": embedding_size,
            "num_codebooks": num_codebooks,
            "codebook_size": codebook_size,
            "item_codes": item_codes,
            "rqvae_tokenizer": rqvae_tokenizer,
            **{k: v for k, v in model_cfg.items()},
        }
        if model_type == "tiger":
            model = TigerRec(**model_kwargs)
        else:
            model = TigerLiteCoGen(**model_kwargs)
        total_params = sum(p.numel() for p in model.parameters())
        be_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith(("_A_u", "_A_v", "_a")))
        if be_params > 0:
            print(f"  Total params: {total_params:,} | Behavioral params: {be_params:,}")
        else:
            print(f"  Total params: {total_params:,}")

        output_dir = Path(args.output_dir) / dataset / f"seed{args.seed}"
        result = train(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            output_dir=str(output_dir),
            max_epochs=training_cfg.get("max_epochs", 200),
            patience=training_cfg.get("patience", 5),
            lr=training_cfg.get("lr", 5e-4),
            weight_decay=training_cfg.get("weight_decay", 0.0),
            device=device,
            seed=args.seed,
        )
        print(f"\nResults saved to {output_dir}/result.json")

    print("Done.")


if __name__ == "__main__":
    main()
