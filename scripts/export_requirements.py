"""Export pinned package versions to a requirements file."""

from __future__ import annotations

import importlib.metadata as metadata
from pathlib import Path
from typing import Iterable, List


def export_requirements(path: str, packages: Iterable[str]) -> None:
    """Write package==version lines for the provided packages."""

    lines: List[str] = []
    for name in packages:
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        lines.append(f\"{name}=={version}\")
    Path(path).write_text(\"\\n\".join(lines) + \"\\n\")


if __name__ == \"__main__\":
    export_requirements(\n        \"requirements.txt\",\n        [\"numpy\", \"pandas\", \"scikit-learn\", \"scipy\", \"rbo\", \"textdistance\"],\n    )
