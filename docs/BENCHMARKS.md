# Benchmarking and Scaling

The table below is an initial closed-form memory estimate for eight heads, head dimension 64, and
bf16/fp16 attention tensors. Runtime measurements should be generated on the target GPU with:

```bash
python scripts/benchmark_throughput.py
python scripts/run_benchmark_suite.py
```

| Sequence Length | Quadratic Attention GiB | Linear Attention GiB | Reduction |
| --- | ---: | ---: | ---: |
| 1,024 | 0.0156 | 0.0029 | 5.3x |
| 4,096 | 0.2500 | 0.0117 | 21.3x |
| 16,384 | 4.0000 | 0.0469 | 85.3x |
| 65,536 | 64.0000 | 0.1875 | 341.3x |
| 1,048,576 | 16,384.0000 | 3.0000 | 5,461.3x |

This is not a substitute for GPU profiling. It is the baseline scaling proof that determines which
sequence lengths are worth running through Nsight, PyTorch profiler, or cluster benchmarks.

## Model-Family Comparisons

`scripts/run_benchmark_suite.py` emits comparison rows for:

- vanilla transformers
- Longformer-style windowed attention
- Performer
- HyenaDNA-style long convolution
- Mamba-style state-space mixing
- DNABERT-2-style bounded-context transformer baselines
