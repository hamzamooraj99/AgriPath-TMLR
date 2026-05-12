from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import imagehash
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from PIL import Image
from torch.utils.data import DataLoader


HARD_DUPLICATE_THRESHOLD = 0.98
SOFT_LEAKAGE_THRESHOLD = 0.95
SEARCH_BATCH_SIZE = 256
BIT_COUNT_LUT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.uint8)


@dataclass
class SplitHashes:
    split_name: str
    hashes: np.ndarray
    files: List[str]
    num_bits: int


class ImageDataset:
    def __init__(self, hf_split: Dataset, split_name: str):
        self.hf_split = hf_split
        self.split_name = split_name

    def __len__(self) -> int:
        return len(self.hf_split)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.hf_split[index]
        image = sample["image"]
        pil_image = self._ensure_pil(image)

        return {
            "image": pil_image,
            "file_path": self._resolve_file_path(sample=sample, image=image, index=index),
            "split": self.split_name,
            "index": index,
        }

    @staticmethod
    def _ensure_pil(image_obj: object) -> Image.Image:
        if isinstance(image_obj, Image.Image):
            return image_obj.convert("RGB")

        if isinstance(image_obj, dict):
            if "path" in image_obj and image_obj["path"] and os.path.exists(image_obj["path"]):
                with Image.open(image_obj["path"]) as img:
                    return img.convert("RGB")

            if "bytes" in image_obj and image_obj["bytes"] is not None:
                from io import BytesIO

                with Image.open(BytesIO(image_obj["bytes"])) as img:
                    return img.convert("RGB")

        raise TypeError(f"Unsupported image payload type: {type(image_obj)!r}")

    def _resolve_file_path(self, sample: Dict[str, object], image: object, index: int) -> str:
        preferred_keys = (
            "file_name",
            "filename",
            "file_path",
            "filepath",
            "path",
            "image_path",
        )
        for key in preferred_keys:
            value = sample.get(key)
            if isinstance(value, str) and value.strip():
                return value

        if isinstance(image, dict):
            image_path = image.get("path")
            if isinstance(image_path, str) and image_path.strip():
                return image_path

        return f"{self.split_name}/sample_{index:06d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect near-duplicate images and train-to-val/test leakage with imagehash.",
    )
    parser.add_argument(
        "--dataset",
        default="hamzamooraj99/AgriPath-LF16-30k",
        help="Hugging Face dataset name or local dataset path.",
    )
    parser.add_argument(
        "--output",
        default="leakage_report.csv",
        help="CSV output path for flagged pairs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Hash extraction batch size.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Accepted for CLI compatibility with the DINOv2 script. Unused here.",
    )
    parser.add_argument(
        "--intra_threshold",
        type=float,
        default=HARD_DUPLICATE_THRESHOLD,
        help="Similarity threshold for intra-split hard duplicates.",
    )
    parser.add_argument(
        "--leakage_threshold",
        type=float,
        default=SOFT_LEAKAGE_THRESHOLD,
        help="Similarity threshold for train-to-validation/test soft leakage.",
    )
    parser.add_argument(
        "--intra_k",
        type=int,
        default=10,
        help="Number of nearest neighbors to inspect for intra-split duplicate search.",
    )
    parser.add_argument(
        "--cross_k",
        type=int,
        default=10,
        help="Number of nearest train neighbors to inspect for validation/test leakage search.",
    )
    parser.add_argument(
        "--hash_method",
        choices=("average", "phash", "dhash", "whash"),
        default="phash",
        help="imagehash algorithm to use.",
    )
    parser.add_argument(
        "--hash_size",
        type=int,
        default=8,
        help="Hash size passed to imagehash.",
    )
    return parser.parse_args()


def collate_fn(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "images": [item["image"] for item in batch],
        "file_paths": [item["file_path"] for item in batch],
        "indices": [item["index"] for item in batch],
        "splits": [item["split"] for item in batch],
    }


def get_hash_fn(method: str) -> Callable[[Image.Image, int], imagehash.ImageHash]:
    mapping = {
        "average": imagehash.average_hash,
        "phash": imagehash.phash,
        "dhash": imagehash.dhash,
        "whash": imagehash.whash,
    }
    return mapping[method]


def hash_to_packed_bits(hash_value: imagehash.ImageHash) -> Tuple[np.ndarray, int]:
    flat_bits = np.asarray(hash_value.hash, dtype=np.uint8).reshape(-1)
    return np.packbits(flat_bits), int(flat_bits.size)


def extract_split_hashes(
    hf_split: Dataset,
    split_name: str,
    hash_fn: Callable[[Image.Image, int], imagehash.ImageHash],
    hash_size: int,
    batch_size: int,
    num_workers: int,
) -> SplitHashes:
    ds = ImageDataset(hf_split=hf_split, split_name=split_name)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    all_hashes: List[np.ndarray] = []
    all_files: List[str] = []
    num_bits: int | None = None

    for batch in loader:
        for image, file_path in zip(batch["images"], batch["file_paths"]):
            packed_hash, current_bits = hash_to_packed_bits(hash_fn(image, hash_size=hash_size))
            if num_bits is None:
                num_bits = current_bits
            elif num_bits != current_bits:
                raise ValueError(
                    f"Inconsistent hash width detected in {split_name}: expected {num_bits} bits, got {current_bits}."
                )
            all_hashes.append(packed_hash.astype(np.uint8, copy=False))
            all_files.append(file_path)

    if not all_hashes:
        packed_width = (hash_size * hash_size + 7) // 8
        return SplitHashes(
            split_name=split_name,
            hashes=np.zeros((0, packed_width), dtype=np.uint8),
            files=[],
            num_bits=hash_size * hash_size,
        )

    return SplitHashes(
        split_name=split_name,
        hashes=np.stack(all_hashes, axis=0),
        files=all_files,
        num_bits=int(num_bits),
    )


def unique_pair_key(
    source_split: str,
    source_file: str,
    match_split: str,
    match_file: str,
) -> Tuple[str, str, str, str]:
    if (source_split, source_file) <= (match_split, match_file):
        return source_split, source_file, match_split, match_file
    return match_split, match_file, source_split, source_file


def batched_hamming_distances(
    queries: np.ndarray,
    references: np.ndarray,
    batch_size: int = SEARCH_BATCH_SIZE,
) -> Tuple[np.ndarray, int]:
    total = len(queries)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        xor = np.bitwise_xor(queries[start:end, None, :], references[None, :, :])
        distances = BIT_COUNT_LUT[xor].sum(axis=2, dtype=np.uint16)
        yield distances, start


def topk_smallest(distances: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    if k >= distances.shape[1]:
        candidate_indices = np.tile(np.arange(distances.shape[1]), (distances.shape[0], 1))
    else:
        candidate_indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]

    candidate_distances = np.take_along_axis(distances, candidate_indices, axis=1)
    order = np.argsort(candidate_distances, axis=1)
    sorted_indices = np.take_along_axis(candidate_indices, order, axis=1)
    sorted_distances = np.take_along_axis(candidate_distances, order, axis=1)
    return sorted_distances, sorted_indices


def distance_to_similarity(distance: int, num_bits: int) -> float:
    return 1.0 - (float(distance) / float(num_bits))


def collect_intra_split_pairs(
    split_data: SplitHashes,
    threshold: float,
    k: int,
) -> List[Dict[str, object]]:
    if len(split_data.hashes) < 2:
        return []

    search_k = min(max(k, 2), len(split_data.hashes))
    rows: List[Dict[str, object]] = []
    seen_pairs = set()

    for distances, offset in batched_hamming_distances(split_data.hashes, split_data.hashes):
        for row_idx in range(distances.shape[0]):
            query_idx = offset + row_idx
            distances[row_idx, query_idx] = np.iinfo(distances.dtype).max

        nearest_distances, nearest_indices = topk_smallest(distances, search_k)

        for row_idx in range(nearest_distances.shape[0]):
            query_idx = offset + row_idx
            for distance, neighbor_idx in zip(nearest_distances[row_idx], nearest_indices[row_idx]):
                neighbor_idx = int(neighbor_idx)
                similarity = distance_to_similarity(int(distance), split_data.num_bits)

                if similarity < threshold:
                    continue

                pair_key = unique_pair_key(
                    split_data.split_name,
                    split_data.files[query_idx],
                    split_data.split_name,
                    split_data.files[neighbor_idx],
                )
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                rows.append(
                    {
                        "source_file": split_data.files[query_idx],
                        "match_file": split_data.files[neighbor_idx],
                        "source_split": split_data.split_name,
                        "match_split": split_data.split_name,
                        "similarity_score": similarity,
                    }
                )

    return rows


def collect_cross_split_pairs(
    query_split: SplitHashes,
    reference_split: SplitHashes,
    threshold: float,
    k: int,
) -> List[Dict[str, object]]:
    if len(query_split.hashes) == 0 or len(reference_split.hashes) == 0:
        return []
    if query_split.num_bits != reference_split.num_bits:
        raise ValueError(
            f"Mismatched hash widths: {query_split.split_name} has {query_split.num_bits} bits, "
            f"{reference_split.split_name} has {reference_split.num_bits} bits."
        )

    search_k = min(max(k, 1), len(reference_split.hashes))
    rows: List[Dict[str, object]] = []
    seen_pairs = set()

    for distances, offset in batched_hamming_distances(query_split.hashes, reference_split.hashes):
        nearest_distances, nearest_indices = topk_smallest(distances, search_k)

        for row_idx in range(nearest_distances.shape[0]):
            query_idx = offset + row_idx
            for distance, neighbor_idx in zip(nearest_distances[row_idx], nearest_indices[row_idx]):
                neighbor_idx = int(neighbor_idx)
                similarity = distance_to_similarity(int(distance), query_split.num_bits)

                if similarity < threshold:
                    continue

                pair_key = (
                    query_split.split_name,
                    query_split.files[query_idx],
                    reference_split.split_name,
                    reference_split.files[neighbor_idx],
                )
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                rows.append(
                    {
                        "source_file": query_split.files[query_idx],
                        "match_file": reference_split.files[neighbor_idx],
                        "source_split": query_split.split_name,
                        "match_split": reference_split.split_name,
                        "similarity_score": similarity,
                    }
                )

    return rows


def load_required_splits(dataset_name: str) -> DatasetDict:
    dataset = load_from_disk(dataset_name) if os.path.isdir(dataset_name) else load_dataset(dataset_name)
    required = ("train", "validation", "test")
    missing = [split for split in required if split not in dataset]
    if missing:
        raise ValueError(f"Dataset is missing required splits: {missing}")
    return dataset


def write_csv(rows: Sequence[Dict[str, object]], output_path: str) -> None:
    fieldnames = [
        "source_file",
        "match_file",
        "source_split",
        "match_split",
        "similarity_score",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (
                item["source_split"],
                item["match_split"],
                -float(item["similarity_score"]),
                item["source_file"],
                item["match_file"],
            ),
        ):
            row = dict(row)
            row["similarity_score"] = f"{float(row['similarity_score']):.6f}"
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    _ = args.device

    print(f"Loading dataset: {args.dataset}")
    dataset = load_required_splits(args.dataset)

    print(f"Loading imagehash method: {args.hash_method} (hash_size={args.hash_size})")
    hash_fn = get_hash_fn(args.hash_method)

    split_hashes: Dict[str, SplitHashes] = {}
    for split_name in ("train", "validation", "test"):
        print(f"Hashing split: {split_name} ({len(dataset[split_name])} images)")
        split_hashes[split_name] = extract_split_hashes(
            hf_split=dataset[split_name],
            split_name=split_name,
            hash_fn=hash_fn,
            hash_size=args.hash_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    intra_rows: List[Dict[str, object]] = []
    for split_name in ("train", "validation", "test"):
        print(f"Searching intra-split hard duplicates in {split_name}")
        intra_rows.extend(
            collect_intra_split_pairs(
                split_data=split_hashes[split_name],
                threshold=args.intra_threshold,
                k=args.intra_k,
            )
        )

    print("Searching train-to-validation leakage")
    val_leakage_rows = collect_cross_split_pairs(
        query_split=split_hashes["validation"],
        reference_split=split_hashes["train"],
        threshold=args.leakage_threshold,
        k=args.cross_k,
    )

    print("Searching train-to-test leakage")
    test_leakage_rows = collect_cross_split_pairs(
        query_split=split_hashes["test"],
        reference_split=split_hashes["train"],
        threshold=args.leakage_threshold,
        k=args.cross_k,
    )

    all_rows = intra_rows + val_leakage_rows + test_leakage_rows
    write_csv(all_rows, args.output)

    print("")
    print(f"Saved report to: {args.output}")
    print("Summary")
    print(f"  train intra-split hard duplicates: {sum(1 for row in intra_rows if row['source_split'] == 'train')}")
    print(f"  validation intra-split hard duplicates: {sum(1 for row in intra_rows if row['source_split'] == 'validation')}")
    print(f"  test intra-split hard duplicates: {sum(1 for row in intra_rows if row['source_split'] == 'test')}")
    print(f"  validation -> train soft leakage pairs: {len(val_leakage_rows)}")
    print(f"  test -> train soft leakage pairs: {len(test_leakage_rows)}")
    print(f"  total flagged pairs: {len(all_rows)}")


if __name__ == "__main__":
    main()
