# Biological Validation

## Implemented Task

The repository includes a reproducible promoter motif classification task. It is intentionally
small and deterministic so the full training path can be run in constrained environments.

```bash
python scripts/prepare_dataset.py --examples 1000 --length 256
python -m hogfm.training.train \
  --train data/processed/promoter/train.jsonl \
  --valid data/processed/promoter/valid.jsonl
```

Metrics emitted:

- accuracy
- precision
- recall
- F1
- AUROC

## External Dataset Targets

The next validation tier should replace or augment the synthetic task with:

- ClinVar variant pathogenicity labels.
- ENCODE promoter/enhancer annotations.
- GTEx tissue-specific expression labels.
- 1000 Genomes population variant panels.
- HG38 reference context extraction.

## Claim Boundary

Synthetic promoter classification proves the software path, not biological generalization. Real
model quality claims require frozen dataset versions, train/validation/test splits, leakage checks,
and reported confidence intervals.
