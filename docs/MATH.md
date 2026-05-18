# Mathematical Notes

## Quadratic Attention Baseline

For batch size `B`, heads `H`, sequence length `N`, and head dimension `D`, vanilla attention forms
an attention matrix:

```text
QK^T in R[B, H, N, N]
```

The attention-score memory term is:

```text
O(B * H * N^2)
```

At one million tokens, even storing the score matrix is not practical.

## Kernelized Linear Attention

The implemented fallback uses a positive feature map:

```text
attention(Q, K, V) = phi(Q) @ (phi(K)^T @ V) / (phi(Q) @ sum(phi(K)))
```

This avoids materializing `N x N` scores:

```text
phi(K)^T @ V in R[B, H, M, D]
```

where `M` is the feature dimension. Memory scales closer to:

```text
O(B * H * N * D + B * H * M * D)
```

## Performer/FAVOR+ Approximation

The model implementation also includes a Performer-style random feature map:

```text
phi(x) = exp(Wx - ||x||^2 / 2) / sqrt(M)
```

where `W` is a fixed random projection matrix per attention head. This is an approximation to
softmax attention and is suitable for benchmarking and research iteration. For production
biological claims, approximation error must be measured against exact attention on short contexts.

## Complexity Validation

Use:

```bash
python scripts/generate_scaling_report.py
python scripts/benchmark_throughput.py
```

The first command generates memory estimates. The second measures actual wall-clock throughput.
