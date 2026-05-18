# Research Report Draft

## Abstract

This project implements a compact research platform for hardware-aware genomic foundation models.
It targets the memory barrier created by quadratic attention over long genomic contexts and
provides a runnable tokenizer, kernelized transformer, biological task, serving API, deployment
configuration, and benchmark tooling.

## Contribution

The current contribution is an integrated experimental system:

1. Genomic tokenization across nucleotide, k-mer, and BPE strategies.
2. Performer-style transformer blocks for non-quadratic long-context modeling.
3. End-to-end training and inference surfaces for promoter classification.
4. Variant log-likelihood scoring and sequence embedding endpoints.
5. Benchmark scripts that separate theoretical memory estimates from runtime measurements.
6. Structured experiment tracking with JSONL, TensorBoard, and optional W&B integration.

## Hypothesis

Kernelized attention can make high-context genomic modeling more accessible by reducing attention
memory from an `N x N` term to a sequence-linear representation while preserving enough long-range
signal for biological tasks.

## Evaluation Plan

| Axis | Measurement |
| --- | --- |
| Memory scaling | GiB estimates and measured peak GPU memory |
| Throughput | tokens/sec across sequence lengths |
| Biological utility | AUROC/F1 on promoter and variant tasks |
| Numerical stability | finite outputs, gradient checks, approximation error |
| Deployment quality | API latency, queue depth, container health checks |

## Limitations

The default model is intentionally small and untrained. The repository should be treated as a
research implementation substrate until trained checkpoints and external dataset results are added.
