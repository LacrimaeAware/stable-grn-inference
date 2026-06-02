"""Inspect local DREAM4 raw files without assuming a gold-standard format."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

import pandas as pd


DEFAULT_ROOT = Path("data/raw/dream4")


def infer_dataset_type(path: Path) -> str:
    """Infer a coarse DREAM4 dataset type from a local filename."""
    name = path.name.lower()
    full_path = path.as_posix().lower()
    parts: list[str] = []

    if re.search(r"size100(?:_|/|$)", full_path):
        parts.append("Size100")
    elif re.search(r"size10(?:_|/|$)", full_path):
        parts.append("Size10")

    if "timeseries" in name:
        parts.append("time series expression")
    elif "multifactorial" in name:
        parts.append("multifactorial expression")
    elif "knockdowns" in name:
        parts.append("knockdown expression")
    elif "knockouts" in name and "dualknockouts" not in name:
        parts.append("knockout expression")
    elif "wildtype" in name:
        parts.append("wildtype expression")
    elif "dualknockouts_indexes" in name:
        parts.append("dual-knockout index pairs")
    else:
        parts.append("unknown")

    return ", ".join(parts)


def detect_delimiter(path: Path) -> str:
    """Return the likely delimiter for a small text file."""
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    counts = Counter({"\t": first_line.count("\t"), ",": first_line.count(",")})
    if counts["\t"] >= counts[","] and counts["\t"] > 0:
        return "tab"
    if counts[","] > 0:
        return "comma"
    return "unknown"


def read_shape(path: Path) -> tuple[int, int]:
    """Read a TSV-like file and return its row/column shape."""
    frame = pd.read_csv(path, sep="\t")
    return frame.shape


def first_rows(path: Path, n: int = 3) -> str:
    """Return the first few parsed rows as a compact Markdown table."""
    frame = pd.read_csv(path, sep="\t")
    return frame.head(n).to_markdown(index=False)


def iter_files(root: Path) -> list[Path]:
    """List local DREAM4 files in a stable order."""
    return sorted(path for path in root.rglob("*") if path.is_file())


def print_inventory(root: Path = DEFAULT_ROOT) -> None:
    """Print an inventory summary for local DREAM4 files."""
    files = iter_files(root)
    gold_candidates = [
        path
        for path in files
        if re.search(r"gold|standard|truth|network|edge", path.name, re.IGNORECASE)
    ]
    print(f"Root: {root}")
    print(f"Files: {len(files)}")
    print(f"Gold-standard filename candidates: {len(gold_candidates)}")
    if gold_candidates:
        for path in gold_candidates:
            print(f"  - {path.as_posix()}")
    print()

    for path in files:
        rel_path = path.as_posix()
        delimiter = detect_delimiter(path)
        shape = read_shape(path)
        print(f"- {rel_path}")
        print(f"  type: {infer_dataset_type(path)}")
        print(f"  delimiter: {delimiter}")
        print(f"  shape: {shape[0]} rows x {shape[1]} columns")


if __name__ == "__main__":
    print_inventory()
