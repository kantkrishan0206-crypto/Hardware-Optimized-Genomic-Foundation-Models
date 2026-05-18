from __future__ import annotations

import argparse
import time

import torch

from hogfm.kernels.linear_attention import estimate_attention_memory, linear_attention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark linear attention token throughput.")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    shape = (args.batch, args.heads, args.tokens, args.dim)
    query = torch.randn(shape, device=device)
    key = torch.randn(shape, device=device)
    value = torch.randn(shape, device=device)

    for _ in range(args.warmup):
        linear_attention(query, key, value)
        if device.type == "cuda":
            torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(args.steps):
        linear_attention(query, key, value)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens_per_second = args.batch * args.tokens * args.steps / elapsed
    estimate = estimate_attention_memory(args.tokens, args.heads, args.dim)
    print(f"device={device}")
    print(f"tokens_per_second={tokens_per_second:.2f}")
    print(f"quadratic_attention_gib={estimate.quadratic_attention_bytes / 1024**3:.4f}")
    print(f"linear_attention_gib={estimate.linear_attention_bytes / 1024**3:.4f}")
    print(f"memory_reduction_factor={estimate.reduction_factor:.2f}")


if __name__ == "__main__":
    main()
