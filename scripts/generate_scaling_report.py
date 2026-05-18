from __future__ import annotations

import argparse
from pathlib import Path


def estimate(tokens: int, heads: int, dim: int, dtype_bytes: int) -> tuple[float, float, float]:
    quadratic = tokens * tokens * heads * dtype_bytes / 1024**3
    linear = tokens * heads * dim * dtype_bytes * 3 / 1024**3
    return quadratic, linear, quadratic / linear


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate attention memory scaling report.")
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096, 16384, 65536, 1048576])
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--output", default="docs/BENCHMARKS.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lines = [
        "# Benchmarking and Scaling",
        "",
        "This table is generated from the closed-form memory model used by the API.",
        "Runtime benchmark CSVs can be produced with `scripts/benchmark_throughput.py`.",
        "",
        "| Sequence Length | Quadratic Attention GiB | Linear Attention GiB | Reduction |",
        "| --- | ---: | ---: | ---: |",
    ]
    for length in args.lengths:
        quadratic, linear, reduction = estimate(length, args.heads, args.dim, args.dtype_bytes)
        lines.append(f"| {length:,} | {quadratic:.4f} | {linear:.4f} | {reduction:.1f}x |")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote={output}")


if __name__ == "__main__":
    main()
