from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class DistributedConfig:
    backend: str = "nccl"
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    resume_from: Path | None = None
    checkpoint_dir: Path = Path("checkpoints/distributed")


def init_distributed(backend: str = "nccl") -> bool:
    if not torch.distributed.is_available():
        return False
    if torch.distributed.is_initialized():
        return True
    rank = int(os.getenv("RANK", "-1"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if rank < 0 or world_size <= 1:
        return False
    torch.distributed.init_process_group(backend=backend)
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))
    return True


def wrap_fsdp(model: torch.nn.Module, config: DistributedConfig) -> torch.nn.Module:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision
    except ImportError as exc:
        raise ImportError("FSDP requires a PyTorch build with distributed support.") from exc
    dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16
    policy = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
    return FSDP(model, mixed_precision=policy)


def save_distributed_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str | Path,
    step: int,
) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "rank": int(os.getenv("RANK", "0")),
    }
    checkpoint = target / f"rank{payload['rank']}_step{step}.pt"
    torch.save(payload, checkpoint)
    return checkpoint


def load_distributed_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str | Path,
) -> int:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"], strict=False)
    optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("step", 0))


def validate_nccl() -> dict[str, int | bool | str]:
    available = torch.distributed.is_available()
    initialized = torch.distributed.is_initialized() if available else False
    return {
        "distributed_available": available,
        "distributed_initialized": initialized,
        "cuda_available": torch.cuda.is_available(),
        "world_size": torch.distributed.get_world_size() if initialized else 1,
        "rank": torch.distributed.get_rank() if initialized else 0,
        "backend": torch.distributed.get_backend() if initialized else "none",
    }
