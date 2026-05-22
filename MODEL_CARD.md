# HOGFM Tiny Demonstration Model Card

## Model Summary

HOGFM is positioned as a hardware-optimized long-context genomic foundation model platform for
scalable biological sequence modeling and distributed genomic AI research.

This model card documents the tiny demonstration checkpoint used to validate the repository's
end-to-end training, checkpointing, tokenizer export, and inference paths. It is not a clinical
model and is not trained on real patient or population-scale genomic data.

## Architecture

- tokenizer: nucleotide tokenizer with special biological tokens
- model: compact Performer-style genomic transformer
- hidden size: 32
- layers: 1
- heads: 4
- objective: masked/next-token-style language modeling demonstration

## Dataset

The tiny checkpoint is trained on a few synthetic DNA strings only to prove the pipeline. Real
research use should train on versioned HG38/ClinVar/ENCODE/GTEx/GenomicBenchmarks datasets.

## Intended Use

- integration testing
- checkpoint loading tests
- Hugging Face export validation
- API smoke tests
- reproducibility demonstrations

## Not Intended For

- clinical decisions
- disease risk prediction
- variant pathogenicity claims
- publication-level biological conclusions

## Safety and Ethics

Clinical genomics requires strict validation, bias analysis, consent-aware data governance, and
expert review. This repository provides infrastructure, not medical advice.
