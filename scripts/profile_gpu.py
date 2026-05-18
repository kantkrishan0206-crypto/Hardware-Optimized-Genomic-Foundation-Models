from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hogfm.kernels.linear_attention import linear_attention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile attention kernels with PyTorch profiler.")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output-dir", default="outputs/profiler")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU profiling.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shape = (1, args.heads, args.tokens, args.dim)
    query = torch.randn(shape, device="cuda")
    key = torch.randn(shape, device="cuda")
    value = torch.randn(shape, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        for _ in range(args.steps):
            linear_attention(query, key, value)
            profiler.step()
    profiler.export_chrome_trace(str(output_dir / "trace.json"))
    summary = {
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "tokens": args.tokens,
        "heads": args.heads,
        "dim": args.dim,
        "steps": args.steps,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
