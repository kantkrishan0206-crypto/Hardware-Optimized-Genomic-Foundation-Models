from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class OptionalDependencyStatus:
    name: str
    available: bool
    install_hint: str


OPTIONAL_DEPENDENCIES: dict[str, str] = {
    "triton": "pip install -r requirements-gpu.txt",
    "flash_attn": "pip install -r requirements-gpu.txt",
    "xformers": "pip install -r requirements-gpu.txt",
    "deepspeed": "pip install -r requirements-gpu.txt",
    "bitsandbytes": "pip install -r requirements-gpu.txt",
    "peft": "pip install -r requirements-gpu.txt",
}


def dependency_status(name: str) -> OptionalDependencyStatus:
    hint = OPTIONAL_DEPENDENCIES.get(name, f"pip install {name}")
    return OptionalDependencyStatus(
        name=name,
        available=importlib.util.find_spec(name) is not None,
        install_hint=hint,
    )


def require_optional(name: str) -> None:
    status = dependency_status(name)
    if not status.available:
        raise ImportError(f"Optional dependency `{name}` is not installed. {status.install_hint}")
