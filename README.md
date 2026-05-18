# Hardware-Optimized Genomic Foundation Models

Research-engineering implementation for long-context genomic sequence modeling with
kernelized attention, reproducible biological tasks, async serving, and deployment scaffolding.

## What Is Implemented

- Genomic tokenizer with nucleotide, k-mer, and simple BPE modes.
- Performer-style genomic transformer with embeddings, positional encoding, transformer blocks,
  language-model loss, sequence classification head, and `loss.backward()` support.
- Linear attention and closed-form memory estimator for comparing `O(N^2)` and `O(N)` scaling.
- Promoter classification task with deterministic dataset generation and AUROC/F1 metrics.
- Training loop with checkpointing for promoter classifiers.
- FastAPI serving layer with `/predict`, `/score_variant`, `/embed_sequence`,
  FASTA/VCF context parsing, and async inference queue.
- Static web research console in `web/`.
- Docker, Compose, Kubernetes, GPU scheduling patch, CI, structured JSON logging, and docs.

## Research Direction

| Rank | Project Direction | Strategic Alignment | Stanford Match | Objective |
| --- | --- | --- | --- | --- |
| 1 | Hardware-Optimized Genomic Foundation Models | Low-level systems, PEFT, hardware-software synthesis | Arc Institute / Brian Hie / Christopher Re | Push genomic context modeling toward multi-megabase windows without quadratic attention memory. |

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

## Project Honesty

This repository is now a runnable research platform, not a completed biological foundation model.
The default API model is randomly initialized unless `HOGFM_CHECKPOINT` points to a trained
checkpoint. Real publication-grade claims require external genomic datasets, GPU benchmark logs,
and trained checkpoints produced from the scripts here.
