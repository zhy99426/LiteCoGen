#!/usr/bin/env python3
"""Preprocess Amazon Reviews 2018 data into inter.json + item.json format.

  - Downloads Amazon 2018 reviews/metadata from UCSD
  - clean_text → normalize_meta_row → has_letter_fields → build_histories_from_rows
    → k_core_filter → remap_to_compact_ids → build_item_json

Usage:
    python data/preprocess.py --dataset Toys_and_Games --output_dir ./data/Toys_and_Games
"""
from __future__ import annotations

import argparse
import gzip
import html
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List
from urllib.request import urlopen

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

AMAZON18_BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2"
DEFAULT_USER_K = 5
DEFAULT_ITEM_K = 5
# 2017-01-01 00:00:00 UTC to 2019-01-01 00:00:00 UTC
DEFAULT_REVIEW_START_TIMESTAMP = 1_483_228_800
DEFAULT_REVIEW_END_TIMESTAMP = 1_546_300_800
LETTER_FIELDS = ("title", "description", "features")

def disable_proxy_env() -> None:
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        os.environ.pop(key, None)

def list_to_str(value) -> str:
    if isinstance(value, list):
        return " ".join(list_to_str(v) for v in value if v is not None)
    if isinstance(value, dict):
        return " ".join(f"{k}:{list_to_str(v)}" for k, v in value.items())
    if value is None:
        return ""
    return str(value)

def clean_text(raw_text) -> str:
    text = list_to_str(raw_text)
    text = html.unescape(text)
    text = text.strip()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[\n\t\r]+", " ", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"[^\x00-\x7F]", " ", text)
    return text.strip()

def normalize_meta_row(obj: dict) -> dict[str, str]:
    features = obj.get("features")
    if features is None:
        features = obj.get("feature", "")
    return {
        "title": clean_text(obj.get("title", "")),
        "description": clean_text(obj.get("description", "")),
        "features": clean_text(features),
    }

def has_letter_fields(meta_row: dict[str, str]) -> bool:
    return any(str(meta_row.get(field, "")).strip() for field in LETTER_FIELDS)

def letter_clean_text(raw_text) -> str:
    """Only used by generate_embeddings.py; kept here for completeness."""
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

def _open_binary(source: str):
    if source.startswith("http://") or source.startswith("https://"):
        return urlopen(source, timeout=120)
    return open(source, "rb")

def iter_json_gz(source: str) -> Iterator[dict]:
    with _open_binary(source) as raw:
        with gzip.GzipFile(fileobj=raw) as gz:
            for line in gz:
                line = line.strip()
                if line:
                    yield json.loads(line)

def review_url(domain: str, base_url: str = AMAZON18_BASE_URL) -> str:
    return f"{base_url}/categoryFilesSmall/{domain}_5.json.gz"

def meta_url(domain: str, base_url: str = AMAZON18_BASE_URL) -> str:
    return f"{base_url}/metaFiles2/meta_{domain}.json.gz"

def _timestamp_in_window(timestamp, start_timestamp: int | None, end_timestamp: int | None) -> bool:
    timestamp = int(timestamp)
    if start_timestamp is not None and timestamp < int(start_timestamp):
        return False
    if end_timestamp is not None and timestamp >= int(end_timestamp):
        return False
    return True

def load_review_rows(
    domain: str,
    base_url: str = AMAZON18_BASE_URL,
    start_timestamp: int | None = DEFAULT_REVIEW_START_TIMESTAMP,
    end_timestamp: int | None = DEFAULT_REVIEW_END_TIMESTAMP,
) -> tuple[list[tuple[str, str, int]], set[str], int]:
    rows: list[tuple[str, str, int]] = []
    items: set[str] = set()
    raw_count = 0

    for obj in iter_json_gz(review_url(domain, base_url)):
        raw_count += 1
        user = obj.get("reviewerID")
        item = obj.get("asin")
        if not user or not item:
            continue
        try:
            timestamp = int(obj.get("unixReviewTime") or 0)
        except (TypeError, ValueError):
            continue
        if not _timestamp_in_window(timestamp, start_timestamp, end_timestamp):
            continue

        raw_item = str(item)
        rows.append((str(user), raw_item, timestamp))
        items.add(raw_item)

    rows.sort(key=lambda value: value[2])
    return rows, items, raw_count

def load_meta_map(domain: str, needed_items: set[str], base_url: str = AMAZON18_BASE_URL) -> tuple[dict[str, dict], int]:
    if not needed_items:
        return {}, 0

    meta_map: dict[str, dict] = {}
    raw_count = 0
    for obj in iter_json_gz(meta_url(domain, base_url)):
        raw_count += 1
        asin = obj.get("asin")
        if not asin:
            continue
        asin = str(asin)
        if asin not in needed_items or asin in meta_map:
            continue
        meta_row = normalize_meta_row(obj)
        if not has_letter_fields(meta_row):
            continue
        meta_map[asin] = meta_row
    return meta_map, raw_count

def build_histories_from_rows(
    rows: Iterable[tuple[str, str, int]],
    meta_item_set: set[str],
    keep_missing_metadata: bool = False,
) -> dict:
    user_raw2id: dict[str, int] = {}
    item_raw2id: dict[str, int] = {}
    user_id2raw: list[str] = []
    item_id2raw: list[str] = []
    user_histories: dict[int, list[int]] = defaultdict(list)
    user_seen_items: dict[int, set[int]] = defaultdict(set)
    missing_meta = 0
    duplicate_pairs = 0

    for raw_user, raw_item, _timestamp in rows:
        if raw_item not in meta_item_set:
            missing_meta += 1
            if not keep_missing_metadata:
                continue

        user_id = user_raw2id.get(raw_user)
        if user_id is None:
            user_id = len(user_raw2id)
            user_raw2id[raw_user] = user_id
            user_id2raw.append(raw_user)

        item_id = item_raw2id.get(raw_item)
        if item_id is None:
            item_id = len(item_raw2id)
            item_raw2id[raw_item] = item_id
            item_id2raw.append(raw_item)

        if item_id in user_seen_items[user_id]:
            duplicate_pairs += 1
            continue

        user_seen_items[user_id].add(item_id)
        user_histories[user_id].append(item_id)

    user_counts = {uid: len(items) for uid, items in user_histories.items()}
    item_counts = Counter()
    for items in user_histories.values():
        item_counts.update(items)

    return {
        "histories": user_histories,
        "user_counts": user_counts,
        "item_counts": item_counts,
        "user_id2raw": user_id2raw,
        "item_id2raw": item_id2raw,
        "missing_meta": missing_meta,
        "duplicate_pairs": duplicate_pairs,
    }

def k_core_filter(
    histories: dict[int, list[int]],
    user_counts: dict[int, int],
    item_counts: Counter,
    user_k: int = DEFAULT_USER_K,
    item_k: int = DEFAULT_ITEM_K,
):
    iteration = 0
    while True:
        valid_users = {uid for uid, count in user_counts.items() if count >= user_k}
        valid_items = {iid for iid, count in item_counts.items() if count >= item_k}

        if len(valid_users) == len(user_counts) and len(valid_items) == len(item_counts):
            break

        iteration += 1
        new_histories: dict[int, list[int]] = {}
        new_user_counts: dict[int, int] = {}
        new_item_counts: Counter = Counter()

        for uid, items in histories.items():
            if uid not in valid_users:
                continue
            filtered_items = [iid for iid in items if iid in valid_items]
            if not filtered_items:
                continue
            new_histories[uid] = filtered_items
            new_user_counts[uid] = len(filtered_items)
            new_item_counts.update(filtered_items)

        histories = new_histories
        user_counts = new_user_counts
        item_counts = new_item_counts
        print(
            f"  k-core iter {iteration}: inters={sum(len(v) for v in histories.values())}, "
            f"users={len(user_counts)}, items={len(item_counts)}"
        )

    return histories, user_counts, item_counts

def remap_to_compact_ids(
    histories: dict[int, list[int]],
    user_id2raw: list[str],
    item_id2raw: list[str],
):
    old_user_to_new: dict[int, int] = {}
    old_item_to_new: dict[int, int] = {}
    inter_json: dict[str, list[int]] = {}

    for old_user_id in histories.keys():
        items = histories[old_user_id]
        if not items:
            continue

        new_user_id = len(old_user_to_new)
        old_user_to_new[old_user_id] = new_user_id

        new_items: list[int] = []
        for old_item_id in items:
            new_item_id = old_item_to_new.get(old_item_id)
            if new_item_id is None:
                new_item_id = len(old_item_to_new)
                old_item_to_new[old_item_id] = new_item_id
            new_items.append(new_item_id)

        inter_json[str(new_user_id)] = new_items

    user_raw2new = {user_id2raw[old_uid]: new_uid for old_uid, new_uid in old_user_to_new.items()}
    item_raw2new = {item_id2raw[old_iid]: new_iid for old_iid, new_iid in old_item_to_new.items()}

    return inter_json, user_raw2new, item_raw2new, old_item_to_new

def build_item_json(meta_map: dict[str, dict], item_raw2new: dict[str, int], item_raw2old=None) -> dict[str, dict]:
    del item_raw2old
    item_json: dict[str, dict] = {}
    for raw_item, new_item_id in sorted(item_raw2new.items(), key=lambda kv: kv[1]):
        item_json[str(new_item_id)] = meta_map.get(raw_item, {"title": "", "description": "", "features": ""})
    return item_json

def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def process_domain(
    domain: str,
    output_root: Path,
    user_k: int = 5,
    item_k: int = 5,
    start_timestamp: int = DEFAULT_REVIEW_START_TIMESTAMP,
    end_timestamp: int = DEFAULT_REVIEW_END_TIMESTAMP,
) -> dict:
    print(f"\n=== {domain} ===")

    domain_dir = output_root / domain
    inter_path = domain_dir / f"{domain}.inter.json"
    item_path = domain_dir / f"{domain}.item.json"

    review_rows, review_item_set, raw_review_count = load_review_rows(
        domain, start_timestamp=start_timestamp, end_timestamp=end_timestamp,
    )
    review_window_count = len(review_rows)
    print(f"loaded review lines: {raw_review_count}")
    print(f"kept 2018 reviews: {review_window_count}")

    if not review_rows:
        result = {
            "dataset": domain, "review_start_timestamp": start_timestamp,
            "review_end_timestamp": end_timestamp,
            "user_k": user_k, "item_k": item_k,
            "raw_review_lines": raw_review_count, "reviews_in_window": review_window_count,
            "user_count": 0, "item_count": 0, "interaction_count": 0,
            "output_dir": str(domain_dir), "inter_path": str(inter_path), "item_path": str(item_path),
            "status": "empty_after_time_filter",
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    meta_map, raw_meta_count = load_meta_map(domain, review_item_set)
    meta_item_set = set(meta_map)
    print(f"loaded meta lines: {raw_meta_count}")
    print(f"kept meta items: {len(meta_map)}")

    built = build_histories_from_rows(review_rows, meta_item_set, keep_missing_metadata=False)
    missing_meta_review_lines = built["missing_meta"]
    duplicate_user_item_lines = built["duplicate_pairs"]
    del review_rows

    histories, _user_counts, _item_counts = k_core_filter(
        built["histories"], built["user_counts"], built["item_counts"],
        user_k=user_k, item_k=item_k,
    )

    if not histories:
        result = {
            "dataset": domain, "review_start_timestamp": start_timestamp,
            "review_end_timestamp": end_timestamp,
            "user_k": user_k, "item_k": item_k,
            "raw_review_lines": raw_review_count, "reviews_in_window": review_window_count,
            "raw_meta_lines": raw_meta_count,
            "missing_meta_review_lines": missing_meta_review_lines,
            "duplicate_user_item_lines": duplicate_user_item_lines,
            "user_count": 0, "item_count": 0, "interaction_count": 0,
            "output_dir": str(domain_dir), "inter_path": str(inter_path), "item_path": str(item_path),
            "status": "empty_after_kcore",
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    inter_json, _user_raw2new, item_raw2new, _old_item_to_new = remap_to_compact_ids(
        histories, built["user_id2raw"], built["item_id2raw"],
    )
    item_json = build_item_json(meta_map, item_raw2new)
    del meta_map

    domain_dir.mkdir(parents=True, exist_ok=True)
    write_json(inter_path, inter_json)
    write_json(item_path, item_json)

    user_count = len(inter_json)
    item_count = len(item_json)
    inter_count = sum(len(v) for v in inter_json.values())
    assert inter_count == sum(len(v) for v in histories.values()), "interaction count mismatch"
    assert set(item_json.keys()) == {str(i) for i in range(item_count)}, "item ids not compact"
    assert set(inter_json.keys()) == {str(i) for i in range(user_count)}, "user ids not compact"

    print(f"  saved: users={user_count}, items={item_count}, inters={inter_count}")

    return {
        "dataset": domain, "review_start_timestamp": start_timestamp,
        "review_end_timestamp": end_timestamp,
        "user_k": user_k, "item_k": item_k,
        "raw_review_lines": raw_review_count, "reviews_in_window": review_window_count,
        "raw_meta_lines": raw_meta_count,
        "missing_meta_review_lines": missing_meta_review_lines,
        "duplicate_user_item_lines": duplicate_user_item_lines,
        "user_count": user_count, "item_count": item_count, "interaction_count": inter_count,
        "output_dir": str(domain_dir), "inter_path": str(inter_path), "item_path": str(item_path),
        "status": "written_no_emb",
    }

# -- CLI --------------------------------------------------------------------

def main():
    disable_proxy_env()

    parser = argparse.ArgumentParser(description="Preprocess Amazon Reviews 2018 data")
    parser.add_argument("--dataset", type=str, default="Toys_and_Games", help="Domain name")
    parser.add_argument("--output_dir", type=str, default="./data/Toys_and_Games", help="Output directory")
    parser.add_argument("--base_url", type=str, default=AMAZON18_BASE_URL)
    parser.add_argument("--user_k", type=int, default=DEFAULT_USER_K)
    parser.add_argument("--item_k", type=int, default=DEFAULT_ITEM_K)
    parser.add_argument("--review_start_timestamp", type=int, default=DEFAULT_REVIEW_START_TIMESTAMP)
    parser.add_argument("--review_end_timestamp", type=int, default=DEFAULT_REVIEW_END_TIMESTAMP)
    args = parser.parse_args()

    output_root = Path(args.output_dir).parent
    result = process_domain(
        args.dataset, output_root,
        user_k=args.user_k, item_k=args.item_k,
        start_timestamp=args.review_start_timestamp,
        end_timestamp=args.review_end_timestamp,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
