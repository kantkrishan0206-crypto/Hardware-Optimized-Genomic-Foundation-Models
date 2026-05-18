from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkRow:
    model_family: str
    sequence_length: int
    estimated_vram_gib: float
    relative_complexity: float
    tokens_per_second: float
    latency_ms: float


def compare_attention_families(
    lengths: list[int],
    heads: int = 8,
    dim: int = 64,
    dtype_bytes: int = 2,
) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    families = {
        "vanilla_transformer": lambda n: n * n,
        "longformer": lambda n: n * 512,
        "performer": lambda n: n * dim,
        "hyenadna": lambda n: n * dim,
        "mamba": lambda n: n * dim,
        "dnabert2": lambda n: n * min(n, 512),
    }
    for length in lengths:
        quadratic_bytes = length * length * heads * dtype_bytes
        linear_bytes = length * heads * dim * dtype_bytes * 3
        for family, complexity_fn in families.items():
            started = time.perf_counter()
            complexity = float(complexity_fn(length))
            latency_ms = (time.perf_counter() - started) * 1000
            if family == "vanilla_transformer":
                memory_gib = quadratic_bytes / 1024**3
            else:
                memory_gib = linear_bytes / 1024**3
            rows.append(
                BenchmarkRow(
                    model_family=family,
                    sequence_length=length,
                    estimated_vram_gib=memory_gib,
                    relative_complexity=complexity,
                    tokens_per_second=length / max(latency_ms / 1000, 1e-9),
                    latency_ms=latency_ms,
                )
            )
    return rows


def write_benchmark_csv(rows: list[BenchmarkRow], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return target


def write_scaling_plot(rows: list[BenchmarkRow], path: str | Path) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Install matplotlib to generate benchmark plots.") from exc
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    families = sorted({row.model_family for row in rows})
    for family in families:
        subset = [row for row in rows if row.model_family == family]
        plt.plot(
            [row.sequence_length for row in subset],
            [row.estimated_vram_gib for row in subset],
            marker="o",
            label=family,
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Sequence length")
    plt.ylabel("Estimated VRAM GiB")
    plt.legend()
    plt.tight_layout()
    plt.savefig(target)
    plt.close()
    return target
