# Master Repository Codegen Blueprint

## Role and Architectural Context

As an expert systems engineer building production-grade infrastructure for genomic foundation models that process exceptionally long DNA contexts.

Generate a repository named `Hardware-Optimized-Genomic-Foundation-Models` that implements a complete open-source scaffold for hardware-aware genomic modeling. The system must prioritize token throughput, memory scaling, and practical training/inference surfaces for long sequences.

## Required Capabilities

1. Parse FASTA and VCF-like variant inputs into validated sequence records.
2. Expose FastAPI endpoints for context extraction and attention-memory estimation.
3. Provide PyTorch linear attention kernels that avoid materializing `N x N` attention maps.
4. Provide clear optional acceleration paths for Triton, CUDA, FlashAttention-3, DeepSpeed, and Hugging Face `peft`.
5. Include runnable tests, benchmark scripts, dependency metadata, Docker packaging, and GitHub Actions CI.

## No-Gap Implementation Rule

Every generated file must contain complete executable code or complete configuration. Do not use ellipses, TODO-only functions, or placeholder-only modules. Optional GPU features may raise explicit dependency errors when the optional package is unavailable, but the CPU/PyTorch fallback path must run.

## Core Files

- `src/hogfm/parsing/fasta.py`
- `src/hogfm/kernels/linear_attention.py`
- `src/hogfm/models/peft_adapter.py`
- `src/hogfm/api/main.py`
- `src/hogfm/training/deepspeed_train.py`
- `scripts/benchmark_throughput.py`
- `configs/deepspeed_zero3.json`
- `configs/model.yaml`
- `tests/test_parsing.py`
- `tests/test_linear_attention.py`
