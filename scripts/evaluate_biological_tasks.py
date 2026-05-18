from __future__ import annotations

import argparse
import json
from pathlib import Path

from hogfm.tasks import (
    brca1_variant_prior,
    enhancer_label,
    evaluate_binary_task,
    splice_site_label,
)


def parse_jsonl(path: Path) -> list[dict[str, str | int | float]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate genomic task predictions from JSONL.")
    parser.add_argument("--task", choices=["enhancer", "splice", "brca1"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/evaluation/metrics.json")
    return parser.parse_args()


def score_row(task: str, row: dict[str, str | int | float]) -> float:
    if task == "enhancer":
        return float(enhancer_label(str(row["sequence"])))
    if task == "splice":
        return float(splice_site_label(str(row["sequence"])))
    if task == "brca1":
        return brca1_variant_prior(str(row["reference"]), str(row["alternate"]))
    raise ValueError(f"Unsupported task: {task}")


def main() -> None:
    args = parse_args()
    rows = parse_jsonl(Path(args.input))
    labels = [int(row["label"]) for row in rows]
    probabilities = [score_row(args.task, row) for row in rows]
    metrics = evaluate_binary_task(labels, probabilities)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
