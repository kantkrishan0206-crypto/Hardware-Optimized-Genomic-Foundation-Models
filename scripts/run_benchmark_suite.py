from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from hogfm.benchmarks import compare_attention_families, write_benchmark_csv
from hogfm.benchmarks.suite import write_scaling_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible HOGFM benchmark suite.")
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096, 16384, 65536])
    parser.add_argument("--csv", default="outputs/benchmarks/model_family_scaling.csv")
    parser.add_argument("--plot", default="outputs/benchmarks/model_family_scaling.png")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = compare_attention_families(args.lengths)
    write_benchmark_csv(rows, args.csv)
    plot_path = None
    if not args.no_plot:
        try:
            plot_path = str(write_scaling_plot(rows, args.plot))
        except ImportError:
            plot_path = "matplotlib_not_installed"
    print(
        json.dumps(
            {
                "csv": args.csv,
                "plot": plot_path,
                "rows": [asdict(row) for row in rows],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
