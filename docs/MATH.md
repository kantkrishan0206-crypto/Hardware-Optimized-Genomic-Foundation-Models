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

## Hyena-Style Operator

The Hyena reference path models long-range interactions with implicit convolutions:

```text
y = gate(x) * conv(value(x), h_theta)
```

where `h_theta` is a learned long filter. This gives long context mixing without pairwise
attention matrices.

## Mamba-Style State Space

The state-space reference path updates a hidden state recurrently:

```text
s_t = exp(delta_t A) * s_{t-1} + B_t x_t
y_t = C_t s_t
```

This offers linear-time sequence processing and is useful as an alternative to attention for
million-token contexts.

## Complexity Validation

Use:

```bash
python scripts/generate_scaling_report.py
python scripts/benchmark_throughput.py
```

The first command generates memory estimates. The second measures actual wall-clock throughput.
