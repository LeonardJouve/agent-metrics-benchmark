"""Chart helpers shared by every experiment: run artifacts any runner writes."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

DEFAULT_DATASET = Path("shared/dataset/sample20.csv")


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def dataset_order(dataset: Path) -> list[str]:
    with dataset.open(encoding="utf-8") as fh:
        return [row["instance_id"] for row in csv.DictReader(fh)]


def load_samples(run_dir: Path) -> tuple[list[float], list[int], list[int], int, float]:
    """(epoch, usage_usec, memory_current, alloc_cpu, alloc_mem_gib) from metrics.jsonl."""
    epochs, usage, mem = [], [], []
    alloc_cpu, alloc_mem = 4, 4.0
    with (run_dir / "metrics.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            s = json.loads(line)
            epochs.append(s["epoch"])
            usage.append(s["usage_usec"])
            mem.append(s["memory_current"])
            alloc_cpu = s.get("alloc_cpu", alloc_cpu)
            alloc_mem = s.get("alloc_mem_gib", alloc_mem)
    return epochs, usage, mem, alloc_cpu, alloc_mem
