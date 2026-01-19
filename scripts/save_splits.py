"""CLI helper to generate and save repeated splits."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tabular_similarity.splits import generate_repeated_splits, save_splits


def process_dataset(path: Path, output_dir: Path) -> None:
    """Generate and save splits for a single dataset."""
    df = pd.read_csv(path)
    output_file = output_dir / f"splits_{path.stem}.json"
    splits = generate_repeated_splits(len(df), n_splits=5, n_repeats=3)
    save_splits(str(output_file), splits)
    print(f"Saved splits for {path.name} -> {output_file}")


def main(data_dir: str, output_dir: str) -> None:
    """Process all CSV datasets in the data directory."""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_files = list(data_path.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return

    for csv_file in csv_files:
        process_dataset(csv_file, output_path)

    print(f"\nProcessed {len(csv_files)} dataset(s)")


if __name__ == "__main__":
    main("data", "results")
