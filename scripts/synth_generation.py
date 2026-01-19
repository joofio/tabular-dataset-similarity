"""Helper to log synthetic generation parameters and invoke an R script."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict


def write_generation_config(path: str, config: Dict) -> None:
    """Write generator parameters to a JSON file for reproducibility."""

    Path(path).write_text(json.dumps(config, indent=2))


def run_r_generator(r_script: str, config_path: str) -> None:
    """Invoke an R script with a config file argument."""

    subprocess.run(["Rscript", r_script, config_path], check=True)


def process_dataset(input_path: Path, output_dir: Path) -> None:
    """Generate config for a single dataset."""
    config = {
        "seed": 20190110,
        "input_path": str(input_path),
        "output_path": str(output_dir / f"synth_{input_path.stem}.csv"),
        "generator": "synthpop",
        "params": {"k": 1000},
    }
    config_path = output_dir / f"synth_config_{input_path.stem}.json"
    write_generation_config(str(config_path), config)
    print(f"Saved config for {input_path.name} -> {config_path}")


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
