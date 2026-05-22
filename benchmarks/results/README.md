# Benchmark Results

This directory stores committed benchmark artifacts so reviewers can see the expected output shape.

Files:

- `model_family_scaling.csv`
- `model_family_scaling.png`
- `benchmark_summary.json`
- `scaling_plot.svg`
- `scaling_plot.png`
- `scaling_report.md`

Hardware-specific plots and profiler traces should be regenerated on the target machine with:

```bash
python scripts/run_benchmark_suite.py --csv benchmarks/results/model_family_scaling.csv --plot benchmarks/results/model_family_scaling.png
python scripts/profile_gpu.py
```
