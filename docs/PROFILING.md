# GPU Profiling

## PyTorch Profiler

```bash
python scripts/profile_gpu.py --tokens 4096 --heads 4 --dim 64
```

Outputs:

- `outputs/profiler/trace.json`
- `outputs/profiler/summary.json`

## Nsight Systems

```bash
nsys profile -o outputs/nsight/hogfm \
  python scripts/profile_gpu.py --tokens 8192 --heads 8 --dim 64
```

## Required Report Fields

- GPU model
- driver version
- CUDA version
- PyTorch version
- peak memory
- kernel timeline
- occupancy notes
- tensor-core eligibility notes
