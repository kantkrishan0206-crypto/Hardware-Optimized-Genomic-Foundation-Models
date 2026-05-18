from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from hogfm.kernels.linear_attention import estimate_attention_memory, linear_attention


def benchmark_linear(
    tokens: int,
    dim: int,
    heads: int,
    batch: int,
    steps: int,
    warmup: int,
    device: torch.device,
) -> dict[str, float | int | str]:
    shape = (batch, heads, tokens, dim)
    query = torch.randn(shape, device=device)
    key = torch.randn(shape, device=device)
    value = torch.randn(shape, device=device)

    for _ in range(warmup):
        linear_attention(query, key, value)
        if device.type == "cuda":
            torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(steps):
        linear_attention(query, key, value)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    estimate = estimate_attention_memory(tokens, heads, dim)
    return {
        "backend": "linear",
        "device": str(device),
        "sequence_length": tokens,
        "heads": heads,
        "dim": dim,
        "batch": batch,
        "seconds": elapsed,
        "tokens_per_second": batch * tokens * steps / elapsed,
        "quadratic_attention_gib": estimate.quadratic_attention_bytes / 1024**3,
        "linear_attention_gib": estimate.linear_attention_bytes / 1024**3,
        "memory_reduction_factor": estimate.reduction_factor,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark genomic attention throughput.")
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/benchmarks/attention_scaling.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    rows = [
        benchmark_linear(
            tokens=length,
            dim=args.dim,
            heads=args.heads,
            batch=args.batch,
            steps=args.steps,
            warmup=args.warmup,
            device=device,
        )
        for length in args.lengths
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    print(f"wrote={output}")


if __name__ == "__main__":
    main()
