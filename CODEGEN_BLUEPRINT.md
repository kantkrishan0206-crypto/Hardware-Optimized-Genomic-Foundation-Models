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

## Stanford/Arc/HazyResearch-Grade Additions

This repository must evolve from a conceptual research scaffold into a fully validated,
benchmarked, biologically evaluated, hardware-optimized genomic foundation model platform
comparable in engineering rigor to HyenaDNA, Evo, DNABERT-2, and other frontier genomic
foundation model repositories.

Required implementation areas:

- million-token genomic contexts with nucleotide-level sequence modeling
- tokenizer systems covering nucleotide, 3-mer, 6-mer, BPE, special biological tokens, and
  Hugging Face-compatible serialization
- Performer/FAVOR+, Hyena-style operators, Mamba-style state-space alternatives, chunked causal
  attention, and FlashAttention compatibility
- Triton/CUDA kernels for fused feature maps, RMSNorm, rotary embeddings, chunking, and memory
  efficient genomic attention
- benchmark suites for VRAM, throughput, latency, scaling graphs, CSV exports, and model-family
  comparisons against vanilla attention, Longformer, HyenaDNA, Mamba, and DNABERT-2
- biological tasks for promoter, enhancer, splice site, pathogenic variant, BRCA1, and ClinVar
- streaming ingestion for FASTA, FASTQ, VCF, BED, and GTF with sharding and HF dataset hooks
- DeepSpeed ZeRO-3, FSDP, NCCL validation, mixed precision, checkpoint resume, and recovery
- W&B, MLflow, TensorBoard, JSONL tracking, Docker, Kubernetes, Helm, cloud templates, and CI/CD
- a novel contribution: adaptive genomic token compression and context routing

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
