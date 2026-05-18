# Hardware-Optimized Genomic Foundation Models

Open-source research engineering scaffold for multi-megabase genomic sequence modeling with hardware-aware attention, PEFT adapters, and distributed training.

## Project Direction

| Rank | Project Direction | Strategic Alignment | Stanford Match | Key Architectural Objective |
| --- | --- | --- | --- | --- |
| 1 | Hardware-Optimized Genomic Foundation Models | Low-level systems, custom PEFT, hardware-software synthesis | Arc Institute / Brian Hie / Christopher Re | Maximize token throughput and eliminate quadratic `O(N^2)` memory scaling during multi-megabase sequence processing. |

## Why This Exists

Genomic token sequences are exceptionally long. A 1-megabase context window processed with standard Transformer attention creates crushing quadratic memory growth, which quickly exhausts GPU memory on real clusters. This repository provides a concrete starting point for:

- Streaming genomic FASTA/variant parsing APIs.
- PyTorch linear attention kernels that scale with sequence length.
- Optional Triton/CUDA and FlashAttention-3 acceleration layers.
- Hugging Face `peft` adapters for genomic foundation models.
- DeepSpeed ZeRO-3 training configuration for memory-constrained multi-GPU execution.
- Throughput benchmarking and CI-ready tests.

## Real-World Impact

The target application surface is high-context genetic risk prediction and variant interpretation, including examples such as BRCA1 locus context modeling or APOE-associated Alzheimer's risk analysis. The engineering goal is to move multi-billion parameter genomic workloads toward accessible compute nodes for localized personalized medicine.

## Repository Layout

```text
.
├── configs/
│   ├── deepspeed_zero3.json
│   └── model.yaml
├── docs/
│   └── ARCHITECTURE.md
├── scripts/
│   └── benchmark_throughput.py
├── src/hogfm/
│   ├── api/main.py
│   ├── kernels/linear_attention.py
│   ├── models/peft_adapter.py
│   ├── parsing/fasta.py
│   └── training/deepspeed_train.py
└── tests/
```

## Quick Start

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

Run the API:

```bash
uvicorn hogfm.api.main:app --reload --host 0.0.0.0 --port 8000
```

Run a small benchmark:

```bash
python scripts/benchmark_throughput.py --tokens 4096 --dim 64 --heads 4
```

## GitHub Push

After creating the remote repository on GitHub, push this local project with:

```bash
git remote add origin https://github.com/kantkrishan0206-crypto/online.git
git push -u origin main
```

If you want the repository name to match the project title exactly, create:

```text
kantkrishan0206-crypto/Hardware-Optimized-Genomic-Foundation-Models
```

and use that URL as the remote instead.
