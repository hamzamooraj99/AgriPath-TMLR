from __future__ import annotations

import argparse
import csv
import os
import re
from typing import Set

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


TEST_SAMPLE_PATTERN = re.compile(r"^test/sample_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove test samples flagged in a leakage report and push the cleaned "
            "AgriPath-LF16-30k dataset to Hugging Face."
        ),
    )
    parser.add_argument(
        "--report",
        default="dataset_scripts/leakage_report_hash.csv",
        help="CSV leakage report to read.",
    )
    parser.add_argument(
        "--dataset",
        default="hamzamooraj99/AgriPath-LF16-30k",
        help="Hugging Face dataset name or local dataset path.",
    )
    parser.add_argument(
        "--target_repo",
        default="hamzamooraj99/AgriPath-LF16-30k-CLEAN",
        help="Destination Hugging Face dataset repo.",
    )
    parser.add_argument(
        "--source_split",
        default="test",
        help="Source split to isolate from the leakage report.",
    )
    parser.add_argument(
        "--match_split",
        default="train",
        help="Match split to isolate from the leakage report.",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="Optional Hugging Face token. Defaults to HF_TOKEN if set.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Push the cleaned dataset as a private repository.",
    )
    return parser.parse_args()


def load_dataset_dict(dataset_name: str) -> DatasetDict:
    dataset = load_from_disk(dataset_name) if os.path.isdir(dataset_name) else load_dataset(dataset_name)
    required = ("train", "validation", "test")
    missing = [split for split in required if split not in dataset]
    if missing:
        raise ValueError(f"Dataset is missing required splits: {missing}")
    return dataset


def extract_flagged_test_indices(
    report_path: str,
    source_split: str,
    match_split: str,
) -> Set[int]:
    flagged_indices: Set[int] = set()

    with open(report_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"source_file", "source_split", "match_split"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Leakage report is missing required columns: {sorted(missing_columns)}")

        for row in reader:
            if row["source_split"] != source_split or row["match_split"] != match_split:
                continue

            sample_ref = row["source_file"].strip()
            match = TEST_SAMPLE_PATTERN.fullmatch(sample_ref)
            if match is None:
                raise ValueError(
                    f"Expected source_file values like 'test/sample_000123' but got {sample_ref!r}."
                )

            flagged_indices.add(int(match.group(1)))

    return flagged_indices


def remove_indices(split: Dataset, indices_to_remove: Set[int]) -> Dataset:
    keep_indices = [index for index in range(len(split)) if index not in indices_to_remove]
    return split.select(keep_indices)


def main() -> None:
    args = parse_args()

    print(f"Reading leakage report: {args.report}")
    flagged_test_indices = extract_flagged_test_indices(
        report_path=args.report,
        source_split=args.source_split,
        match_split=args.match_split,
    )
    print(f"Found {len(flagged_test_indices)} unique flagged test samples to remove")

    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset_dict(args.dataset)

    original_test_size = len(dataset["test"])
    out_of_range = [index for index in sorted(flagged_test_indices) if index >= original_test_size]
    if out_of_range:
        raise ValueError(
            f"Found {len(out_of_range)} flagged indices outside the test split size of {original_test_size}. "
            f"First invalid index: {out_of_range[0]}"
        )

    cleaned_test = remove_indices(dataset["test"], flagged_test_indices)
    cleaned_dataset = DatasetDict(
        {
            "train": dataset["train"],
            "validation": dataset["validation"],
            "test": cleaned_test,
        }
    )

    print("Dataset summary")
    print(f"  train: {len(cleaned_dataset['train'])}")
    print(f"  validation: {len(cleaned_dataset['validation'])}")
    print(f"  test: {len(cleaned_dataset['test'])} (removed {original_test_size - len(cleaned_test)})")

    print(f"Pushing cleaned dataset to: {args.target_repo}")
    push_kwargs = {}
    if args.token:
        push_kwargs["token"] = args.token
    if args.private:
        push_kwargs["private"] = True

    cleaned_dataset.push_to_hub(args.target_repo, **push_kwargs)
    print("Push complete")


if __name__ == "__main__":
    main()
