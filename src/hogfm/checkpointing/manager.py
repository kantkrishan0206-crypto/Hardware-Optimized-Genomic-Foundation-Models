from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class CheckpointMetadata:
    step: int
    epoch: int
    metrics: dict[str, float]
    model_config: dict[str, Any]
    created_at: int


def save_checkpoint(
    model: torch.nn.Module,
    output_dir: str | Path,
    metadata: CheckpointMetadata,
    optimizer: torch.optim.Optimizer | None = None,
    use_safetensors: bool = True,
) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if use_safetensors:
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise ImportError("Install safetensors to save safetensors checkpoints.") from exc
        target = path / "model.safetensors"
        save_file(state_dict, target)
    else:
        target = path / "model.pt"
        torch.save({"model_state_dict": state_dict}, target)
    if optimizer is not None:
        torch.save(optimizer.state_dict(), path / "optimizer.pt")
    (path / "metadata.json").write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")
    return target


def load_checkpoint(model: torch.nn.Module, path: str | Path, strict: bool = False) -> dict[str, Any]:
    target = Path(path)
    if target.is_dir():
        safetensors_path = target / "model.safetensors"
        target = safetensors_path if safetensors_path.exists() else target / "model.pt"
    if target.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("Install safetensors to load safetensors checkpoints.") from exc
        state_dict = load_file(target)
    else:
        payload = torch.load(target, map_location="cpu")
        state_dict = payload.get("model_state_dict", payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return {"missing_keys": list(missing), "unexpected_keys": list(unexpected)}


def export_huggingface_layout(
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: str | Path,
    model_config: dict[str, Any],
) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "pytorch_model.bin")
    (path / "config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(path)
    return path


def merge_lora_if_available(model: Any) -> Any:
    if hasattr(model, "merge_and_unload"):
        return model.merge_and_unload()
    return model


def build_metadata(step: int, epoch: int, metrics: dict[str, float], config: dict[str, Any]) -> CheckpointMetadata:
    return CheckpointMetadata(
        step=step,
        epoch=epoch,
        metrics=metrics,
        model_config=config,
        created_at=int(time.time()),
    )
