from __future__ import annotations

import argparse
import json
from pathlib import Path

from hogfm.tasks.promoter import generate_synthetic_promoter_dataset


def write_jsonl(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare reproducible genomic task datasets.")
    parser.add_argument("--task", choices=["synthetic-promoter"], default="synthetic-promoter")
    parser.add_argument("--examples", type=int, default=1000)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", default="data/processed/promoter")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.valid_fraction < 1.0:
        raise ValueError("--valid-fraction must be between 0 and 1.")
    examples = generate_synthetic_promoter_dataset(
        examples=args.examples,
        length=args.length,
        seed=args.seed,
    )
    split_at = int(round(len(examples) * (1.0 - args.valid_fraction)))
    rows = [
        {"sequence": example.sequence, "label": example.label, "source": example.source}
        for example in examples
    ]
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", rows[:split_at])
    write_jsonl(output_dir / "valid.jsonl", rows[split_at:])
    metadata = {
        "task": args.task,
        "examples": args.examples,
        "length": args.length,
        "seed": args.seed,
        "train_examples": split_at,
        "valid_examples": len(rows) - split_at,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
