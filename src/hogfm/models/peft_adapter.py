from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LoraAdapterConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = field(default_factory=lambda: ("query", "key", "value", "dense"))
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


def _import_peft() -> tuple[Any, Any]:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "PEFT support requires the optional GPU stack. Install with: "
            'pip install "hardware-optimized-genomic-foundation-models[gpu]"'
        ) from exc
    return LoraConfig, get_peft_model


def apply_lora_adapter(model: Any, config: LoraAdapterConfig) -> Any:
    LoraConfig, get_peft_model = _import_peft()
    peft_config = LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        bias=config.bias,
        task_type=config.task_type,
    )
    return get_peft_model(model, peft_config)


def save_adapter(model: Any, output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    if not hasattr(model, "save_pretrained"):
        raise TypeError("Model does not expose save_pretrained; cannot save PEFT adapter.")
    model.save_pretrained(path, safe_serialization=True)
    return path
