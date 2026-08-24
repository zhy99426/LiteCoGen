"""Generate item text embeddings for RQ-VAE input.

  build_letter_item_text_list → _word_drop → generate_item_embeddings → save_embeddings

Uses huggyllama/llama-7b with 3-field encoding (title/description/features),
mean pooling with attention mask, and field averaging.
Produces {dataset}.emb-llama-td.npy.

Usage:
    python data/generate_embeddings.py --dataset Toys_and_Games --data_dir ./data --device cuda:0
"""
from __future__ import annotations

import argparse
import html
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

LETTER_FIELDS = ("title", "description", "features")
MAX_SENT_LEN = 2048
EMB_BATCH_SIZE = 1
PLM_CHECKPOINT = "huggyllama/llama-7b"
PLM_NAME = "llama"
WORD_DROP_RATIO = -1.0

def letter_clean_text(raw_text) -> str:
    """Aggressive clean for embedding input. Character-by-character identical to reference implementation."""
    if raw_text is None:
        raw_text = ""

    if isinstance(raw_text, list):
        cleaned_parts = []
        for raw in raw_text:
            raw = html.unescape(str(raw))
            raw = re.sub(r"</?\w+[^>]*>", "", raw)
            raw = re.sub(r'["\n\r]*', "", raw)
            cleaned_parts.append(raw.strip())
        cleaned_text = " ".join(cleaned_parts)
    else:
        if isinstance(raw_text, dict):
            cleaned_text = str(raw_text)[1:-1].strip()
        else:
            cleaned_text = str(raw_text).strip()
        cleaned_text = html.unescape(cleaned_text)
        cleaned_text = re.sub(r"</?\w+[^>]*>", "", cleaned_text)
        cleaned_text = re.sub(r'["\n\r]*', "", cleaned_text)

    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == ".":
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + "."
    else:
        cleaned_text = cleaned_text[:index] + "."
    if len(cleaned_text) >= 2000:
        cleaned_text = ""
    return cleaned_text

def build_letter_item_text_list(item_json: dict[str, dict]) -> list[tuple[int, list[str]]]:
    """Convert item.json to (item_id, [cleaned_title, cleaned_desc, cleaned_features]) list."""
    item_text_list = []
    for item_id_str in sorted(item_json.keys(), key=int):
        data = item_json[item_id_str]
        item_text_list.append(
            (
                int(item_id_str),
                [letter_clean_text(data.get(field, "")) for field in LETTER_FIELDS],
            )
        )
    return item_text_list

def _word_drop(sentences: list[str], word_drop_ratio: float) -> list[str]:
    if word_drop_ratio <= 0:
        return sentences
    dropped = []
    for sentence in sentences:
        words = sentence.split(" ")
        kept = [word for word in words if random.random() > word_drop_ratio]
        dropped.append(" ".join(kept))
    return dropped

def generate_item_embeddings(
    item_text_list,
    tokenizer,
    model,
    device: torch.device,
    max_sent_len: int = MAX_SENT_LEN,
    batch_size: int = EMB_BATCH_SIZE,
    word_drop_ratio: float = WORD_DROP_RATIO,
    progress_desc: str = "embed items",
    progress_position: int = 0,
) -> np.ndarray:
    """Generate item embeddings via mean pooling over Llama-7B hidden states.

    For each item: encode title, description, features separately,
    mean-pool over token dimension with attention mask, average
    across the 3 fields.
    """
    if not item_text_list:
        hidden_size = getattr(getattr(model, "config", None), "hidden_size", 0)
        return np.empty((0, hidden_size), dtype=np.float32)

    order_texts = [None] * len(item_text_list)
    for item, text in item_text_list:
        order_texts[item] = text
    if any(text is None for text in order_texts):
        raise ValueError("item ids must be compact and contiguous before embedding")

    embeddings = []
    batch_size = max(1, int(batch_size))
    total_batches = (len(order_texts) + batch_size - 1) // batch_size
    model.eval()
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(order_texts), batch_size),
            total=total_batches,
            desc=progress_desc,
            unit="batch",
            position=max(0, int(progress_position)),
            dynamic_ncols=True,
            mininterval=2.0,
            leave=True,
        ):
            batch_texts = order_texts[start : start + batch_size]
            field_count = len(batch_texts[0])
            for item_texts in batch_texts:
                if len(item_texts) != field_count:
                    raise ValueError("all item text rows must contain the same number of fields")

            flat_sentences = [sentence for field_texts in zip(*batch_texts) for sentence in field_texts]
            flat_sentences = _word_drop(flat_sentences, word_drop_ratio)
            encoded = tokenizer(
                flat_sentences,
                max_length=max_sent_len,
                truncation=True,
                return_tensors="pt",
                padding="longest",
            ).to(device)
            outputs = model(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
            )
            attention_mask = encoded.attention_mask
            masked_output = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
            denom = attention_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            mean_output = masked_output.sum(dim=1) / denom
            batch_embedding = mean_output.reshape(field_count, len(batch_texts), -1).mean(dim=0)
            embeddings.append(batch_embedding.detach().cpu())

    return torch.cat(embeddings, dim=0).to(torch.float32).numpy()

def save_embeddings(path: Path, embeddings: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings.astype(np.float32, copy=False))

def main():
    parser = argparse.ArgumentParser(description="Generate item embeddings for RQ-VAE input")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. Toys_and_Games)")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data root directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--plm_checkpoint", type=str, default=PLM_CHECKPOINT)
    parser.add_argument("--plm_name", type=str, default=PLM_NAME)
    parser.add_argument("--max_sent_len", type=int, default=MAX_SENT_LEN)
    parser.add_argument("--batch_size", type=int, default=EMB_BATCH_SIZE)
    parser.add_argument("--word_drop_ratio", type=float, default=WORD_DROP_RATIO)
    args = parser.parse_args()

    data_dir = Path(args.data_dir) / args.dataset
    item_path = data_dir / f"{args.dataset}.item.json"
    emb_path = data_dir / f"{args.dataset}.emb-{args.plm_name}-td.npy"

    if not item_path.exists():
        raise FileNotFoundError(f"item.json not found: {item_path}. Run data/preprocess.py first.")

    print(f"Loading item metadata from {item_path}")
    with open(item_path, "r", encoding="utf-8") as f:
        item_json = json.load(f)

    item_count = len(item_json)
    print(f"  Items: {item_count}")

    item_text_list = build_letter_item_text_list(item_json)

    print(f"Loading model: {args.plm_checkpoint}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.plm_checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    model = AutoModel.from_pretrained(args.plm_checkpoint, low_cpu_mem_usage=True)
    model = model.to(device)
    model.eval()

    batch_size = max(1, int(args.batch_size))
    batch_count = (item_count + batch_size - 1) // batch_size
    print(
        f"  embedding: items={item_count}, batches={batch_count}, batch_size={batch_size}",
        flush=True,
    )
    embeddings = generate_item_embeddings(
        item_text_list,
        tokenizer,
        model,
        device=device,
        max_sent_len=args.max_sent_len,
        batch_size=batch_size,
        word_drop_ratio=args.word_drop_ratio,
        progress_desc=f"{args.dataset} emb",
    )

    assert embeddings.shape[0] == item_count, f"Expected {item_count} rows, got {embeddings.shape[0]}"
    hidden_size = getattr(getattr(model, "config", None), "hidden_size", embeddings.shape[1])
    assert embeddings.shape[1] == hidden_size, f"Expected {hidden_size} dims, got {embeddings.shape[1]}"

    save_embeddings(emb_path, embeddings)
    print(f"Saved: {emb_path}  shape={embeddings.shape}")

    # Also save meta for RQ-VAE cache loader compatibility
    meta_path = data_dir / f"{args.dataset}.semid-rqvae-k256-m3.meta.json"
    meta = {
        "embedding_raw_dim": int(hidden_size),
        "embedding_used_dim": int(hidden_size),
        "rqvae_code_dim": 32,
        "rqvae_hidden_size": 512,
        "rqvae_kmeans_iters": 100,
        "plm_checkpoint": args.plm_checkpoint,
    }
    if not meta_path.exists():
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        print(f"  Meta saved: {meta_path}")

if __name__ == "__main__":
    main()
