# Hardware-Optimized Genomic Foundation Models

Research-engineering implementation for long-context genomic sequence modeling with
kernelized attention, reproducible biological tasks, async serving, and deployment scaffolding.

This repository must evolve from a conceptual research scaffold into a fully validated,
benchmarked, biologically evaluated, hardware-optimized genomic foundation model platform
comparable in engineering rigor to HyenaDNA, Evo, DNABERT-2, and other frontier genomic
foundation model repositories.

## What Is Implemented

- Genomic tokenizer with nucleotide, k-mer, and simple BPE modes.
- Hugging Face-compatible tokenizer serialization artifacts.
- Performer-style genomic transformer with embeddings, positional encoding, transformer blocks,
  language-model loss, sequence classification head, and `loss.backward()` support.
- Linear, Performer/FAVOR+, Hyena-style, Mamba-style, and chunked causal attention paths.
- Optional FlashAttention and Triton integration modules for GPU experiments.
- Promoter, enhancer, splice, BRCA1, and ClinVar-oriented task utilities.
- FASTA, FASTQ, VCF, BED, and GTF streaming dataset ingestion.
- Training loop with checkpointing and JSONL/TensorBoard/optional W&B/MLflow tracking.
- FastAPI serving layer with `/predict`, `/score_variant`, `/embed_sequence`,
  `/generate`, `/benchmark`, FASTA/VCF context parsing, auth, streaming, and async queueing.
- Static web research console in `web/`.
- Docker, Compose, Kubernetes, Helm, cloud templates, GPU scheduling patch, CI, and docs.
- Adaptive genomic token compression as a concrete novel research contribution.

## Research Direction

| Rank | Project Direction | Strategic Alignment | Stanford Match | Objective |
| --- | --- | --- | --- | --- |
| 1 | Hardware-Optimized Genomic Foundation Models | Low-level systems, custom PEFT, hardware-software synthesis | Arc Institute / Brian Hie / Christopher Re | Maximize token throughput and eliminate quadratic `O(N^2)` memory scaling during multi-megabase sequence processing. |

## Why This Exists

Genomic token sequences are exceptionally long. A 1-megabase context window processed with standard
Transformer attention creates crushing quadratic memory growth, which quickly exhausts GPU memory
on real clusters. This repository provides a concrete starting point for:

- Streaming genomic FASTA/variant parsing APIs.
- PyTorch linear attention kernels that scale with sequence length.
- Optional Triton/CUDA and FlashAttention acceleration layers.
- Hugging Face `peft` adapters for genomic foundation models.
- DeepSpeed ZeRO-3 training configuration for memory-constrained multi-GPU execution.
- Throughput benchmarking and CI-ready tests.

## Real-World Impact

The target application surface is high-context genetic risk prediction and variant interpretation,
including examples such as BRCA1 locus context modeling or APOE-associated Alzheimer's risk
analysis. The engineering goal is to move multi-billion parameter genomic workloads toward
accessible compute nodes for localized personalized medicine.

## Repository Layout

```text
configs/      model, DeepSpeed, and tokenizer settings
docs/         architecture, math, deployment, validation, and report drafts
helm/         Helm chart with autoscaling and GPU scheduling knobs
k8s/          Kubernetes deployment and GPU scheduling patch
scripts/      dataset preparation and attention benchmarking
src/hogfm/    tokenizer, model, kernels, tasks, API, training, observability
tests/        parser, tokenizer, model, task, and kernel tests
web/          static research console
```

## Quick Start

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -c constraints.txt -e ".[dev]"
pytest
```

Prepare a reproducible promoter dataset:

```bash
python scripts/prepare_dataset.py --examples 1000 --length 256
```

Train the classifier:

```bash
python -m hogfm.training.train `
  --train data/processed/promoter/train.jsonl `
  --valid data/processed/promoter/valid.jsonl `
  --epochs 3
```

Run the API:

```bash
uvicorn hogfm.api.main:app --host 0.0.0.0 --port 8000
```

Open the static research console:

```text
web/index.html
```

## API Surface

```text
GET  /health
POST /predict
POST /score_variant
POST /embed_sequence
POST /generate
POST /benchmark
POST /api/parse/fasta
POST /api/variant/context
POST /api/variant/context-from-vcf
POST /api/estimate-memory
```

## Benchmarking

Generate theoretical scaling tables:

```bash
python scripts/generate_scaling_report.py
```

Run runtime attention benchmarks:

```bash
python scripts/benchmark_throughput.py --lengths 1024 2048 4096 8192
```

## Production Deployment

```bash
docker compose up --build
kubectl apply -f k8s/deployment.yaml
```

For GPU clusters, apply the scheduling patch after labeling the node pool:

```bash
kubectl patch deployment hogfm-api --patch-file k8s/gpu-nodepool-patch.yaml
```

Helm:

```bash
helm install hogfm helm/hogfm --set gpu.enabled=true
```

## Project Honesty

This repository is now a runnable research platform, not a completed biological foundation model.
The default API model is randomly initialized unless `HOGFM_CHECKPOINT` points to a trained
checkpoint. Real publication-grade claims require external genomic datasets, GPU benchmark logs,
and trained checkpoints produced from the scripts here.
