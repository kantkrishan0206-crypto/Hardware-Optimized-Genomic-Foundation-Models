# GPU Validation Report

## Validation Run

- Date: 2026-05-22
- Host OS: Windows 11
- Python: 3.12.13
- PyTorch: 2.5.1+cu121
- PyTorch CUDA runtime: 12.1
- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU
- NVIDIA driver: 581.95
- Reported CUDA driver capability from `nvidia-smi`: 13.0
- VRAM: 4096 MiB

## Commands Executed

```bash
python scripts/validate_environment.py
pytest -m "not gpu"
pytest -m gpu
python scripts/profile_gpu.py --tokens 256 --heads 2 --dim 32 --steps 1 --output-dir validation/gpu_profiler
```

## Results

| Check | Result |
| --- | --- |
| Environment probe | `validation/environment.json` generated successfully |
| PyTorch CUDA | `torch.cuda.is_available() == True` |
| CPU tests | `27 passed, 2 deselected` |
| GPU tests | `1 passed, 1 skipped` |
| Profiler smoke | `validation/gpu_profiler/summary.json` generated |
| Peak profiler allocation | 8,989,184 bytes |

## Optional GPU Packages

The local Windows validation environment successfully installed and executed CUDA-enabled PyTorch.
Linux-only accelerator packages were intentionally gated by `requirements-gpu.txt` environment
markers or optional import checks:

| Package | Local Status | Reason |
| --- | --- | --- |
| Triton | not imported | Linux CUDA runner required |
| FlashAttention | not imported | Linux CUDA runner required |
| DeepSpeed | not imported | Linux CUDA runner required |
| xFormers | not imported | Linux CUDA runner required |
| bitsandbytes | not imported | Linux CUDA runner required |

This means the repository is CPU-valid and PyTorch-CUDA validated on this machine, while
Triton/FlashAttention/DeepSpeed claims remain gated for the self-hosted Linux GPU workflow.

## Validation Policy

GPU tests are marked with `@pytest.mark.gpu` and excluded from CPU CI with:

```bash
pytest -m "not gpu"
```

Full GPU validation on a Linux CUDA runner should run:

```bash
pip install -r requirements-gpu.txt
pytest -m gpu
python scripts/profile_gpu.py --tokens 4096 --heads 4 --dim 64 --steps 5
```

## Evidence Files

- `validation/environment.json`
- `validation/gpu_profiler/summary.json`
- `validation/gpu_profiler/trace.json`
- `benchmarks/results/model_family_scaling.csv`
- `benchmarks/results/model_family_scaling.png`
