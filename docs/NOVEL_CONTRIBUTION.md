# Novel Contribution: Adaptive Genomic Token Compression

The repository now includes a concrete research contribution in
`hogfm.research.adaptive_compression`.

## Idea

Long genomic contexts contain a mixture of motif-rich regions and low-information repetitive
regions. The adaptive compression pass keeps motif-rich regions at base resolution while replacing
homopolymer runs or extreme-GC windows with summary tokens before attention.

## Why It Matters

This attacks the million-token context problem before the model sees the sequence:

```text
raw bases -> adaptive compression -> routed context windows -> model
```

The mechanism can be combined with Performer/Hyena/Mamba mixers and hardware-aware scheduling.

## Research Hypothesis

Biologically informed compression can lower effective sequence length without erasing functional
signals, improving throughput while preserving downstream task AUROC.

## Evaluation

Future experiments should report:

- compression ratio by chromosome and locus
- task AUROC before and after compression
- tokens/sec improvement
- peak GPU memory reduction
- error analysis on promoter/enhancer/splice loci
