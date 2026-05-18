# Reproducibility

## Dependency Pinning

Use `constraints.txt` for CI and local installs:

```bash
pip install -c constraints.txt -e ".[dev]"
```

## Determinism

Training scripts set Python, NumPy, and PyTorch seeds. Dataset generation also accepts an explicit
seed and writes metadata alongside generated JSONL files.

## Versioned Data

Dataset manifests should include:

- source URL
- checksum
- genome build
- task split hash
- preprocessing command

The synthetic promoter generator already writes `metadata.json`.

## Benchmark Reproduction

```bash
python scripts/generate_scaling_report.py
python scripts/run_benchmark_suite.py --no-plot
```

Runtime GPU reports must include:

- GPU model and driver
- CUDA version
- PyTorch version
- batch size
- precision
- sequence lengths
