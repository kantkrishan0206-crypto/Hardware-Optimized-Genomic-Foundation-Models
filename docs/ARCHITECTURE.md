# Architecture

## Kernel Layer

The kernel layer provides a PyTorch linear attention implementation that avoids allocating an `N x N` attention score matrix. This is the baseline correctness path. Triton, CUDA, and FlashAttention-3 extensions can be added behind the same `attention_forward` shape contract.

## Runtime Layer

PEFT adapters are isolated in `hogfm.models.peft_adapter`, so the base model can remain frozen while LoRA modules capture task-specific genomic behavior. Safetensors-compatible adapter saving is used by default where the backing model supports `save_pretrained`.

## Distributed Layer

`configs/deepspeed_zero3.json` uses ZeRO-3 with CPU optimizer and parameter offload, targeting workloads where context length is the immediate memory pressure. Training code is explicit about optional dependencies so CPU-only parser and API workflows remain usable.

## Application Layer

The FastAPI service exposes:

- FASTA summary parsing.
- Direct variant context extraction.
- VCF-like variant context extraction.
- Attention memory estimation for comparing quadratic and linear scaling.
