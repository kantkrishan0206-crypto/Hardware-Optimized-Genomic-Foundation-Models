# Architecture

## System Flow

```mermaid
flowchart LR
  Web["Static research console"] --> API["FastAPI async queue"]
  API --> Tok["Genomic tokenizer"]
  Tok --> Model["Performer genomic transformer"]
  Model --> Tasks["Promoter / variant / embedding heads"]
  API --> Logs["JSON logs"]
  Bench["Benchmark scripts"] --> Reports["Scaling reports"]
```

## Model Pipeline

```mermaid
flowchart TB
  Seq["DNA sequence"] --> Tokenizer["Nucleotide / k-mer / BPE tokenizer"]
  Tokenizer --> Emb["Token embeddings + sinusoidal positions"]
  Emb --> Blocks["Transformer blocks"]
  Blocks --> Attn["Performer or linear attention"]
  Blocks --> LM["Language modeling logits"]
  Blocks --> Cls["Mean pooling + classifier"]
```

The transformer path is intentionally compact so it can run in CI and on laptops. The same
interfaces are designed for larger hidden sizes, more layers, and GPU acceleration.

## Kernel Acceleration Hooks

- `hogfm.kernels.linear_attention` is the portable PyTorch correctness path.
- `hogfm.kernels.flash_attention` wraps FlashAttention for exact attention baselines on GPU.
- `hogfm.kernels.triton_ops` provides a fused ELU+1 feature-map kernel for CUDA tensors.

These modules fail loudly when optional GPU dependencies are absent, which keeps CPU development
honest while preserving a clear path for hardware-specific experiments.

## Distributed Topology

```mermaid
flowchart LR
  Rank0["Rank 0"] <-->|NCCL| Rank1["Rank 1"]
  Rank1 <-->|NCCL| Rank2["Rank 2"]
  Rank2 <-->|NCCL| Rank3["Rank 3"]
  Rank0 --> ZeRO["DeepSpeed ZeRO-3 optimizer/parameter partitioning"]
  Rank1 --> ZeRO
  Rank2 --> ZeRO
  Rank3 --> ZeRO
```

The repository includes `configs/deepspeed_zero3.json` for large-model experiments. The local
training loop is deliberately dependency-light; the DeepSpeed entry point remains available for
multi-GPU extension.

## Serving Layer

The API owns a bounded async queue. Endpoints enqueue work and a background worker performs model
execution in a thread so long-running sequence calls do not block the ASGI event loop.

## Observability

Application logs are emitted as JSON. Request path, method, latency, and status code are attached
with stable keys for ingestion by Loki, CloudWatch, or managed OpenTelemetry collectors.
