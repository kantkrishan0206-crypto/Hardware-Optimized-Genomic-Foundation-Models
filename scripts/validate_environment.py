from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path


def module_available(name: str) -> bool:
    try:
        __import__(name)
    except ImportError:
        return False
    return True


def nvidia_smi() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return f"nvidia-smi unavailable: {exc}"
    return result.stdout


def main() -> None:
    payload: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "modules": {
            "torch": module_available("torch"),
            "pytest": module_available("pytest"),
            "triton": module_available("triton"),
            "flash_attn": module_available("flash_attn"),
            "deepspeed": module_available("deepspeed"),
            "xformers": module_available("xformers"),
            "bitsandbytes": module_available("bitsandbytes"),
        },
        "nvidia_smi": nvidia_smi(),
    }
    if module_available("torch"):
        import torch

        payload["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    output = Path("validation/environment.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
