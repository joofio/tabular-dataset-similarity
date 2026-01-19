"""Persist experiment metadata and seeds for reproducibility."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def write_metadata(path: str, metadata: Dict) -> None:
    """Write metadata as JSON."""

    Path(path).write_text(json.dumps(metadata, indent=2))


def process_dataset(input_path: Path, output_dir: Path) -> None:
    """Generate metadata for a single dataset."""
    metadata = {
        "random_seed": 42,
        "test_size": 0.2,
        "splits_path": str(output_dir / f"splits_{input_path.stem}.json"),
        "config_path": "configs/experiment.py",
        "dataset": input_path.name,
    }
    output_file = output_dir / f"metadata_{input_path.stem}.json"
    write_metadata(str(output_file), metadata)
    print(f"Saved metadata for {input_path.name} -> {output_file}")


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
