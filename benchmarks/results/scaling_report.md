# Benchmarking and Scaling

This table is generated from the closed-form memory model used by the API.
Runtime benchmark CSVs can be produced with `scripts/benchmark_throughput.py`.

| Sequence Length | Quadratic Attention GiB | Linear Attention GiB | Reduction |
| --- | ---: | ---: | ---: |
| 1,024 | 0.0156 | 0.0029 | 5.3x |
| 4,096 | 0.2500 | 0.0117 | 21.3x |
| 16,384 | 4.0000 | 0.0469 | 85.3x |
| 65,536 | 64.0000 | 0.1875 | 341.3x |
| 1,048,576 | 16384.0000 | 3.0000 | 5461.3x |
