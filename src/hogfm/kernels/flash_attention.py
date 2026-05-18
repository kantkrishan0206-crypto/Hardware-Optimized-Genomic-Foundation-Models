from __future__ import annotations

import torch


def flash_attention_available() -> bool:
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


def flash_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Run FlashAttention when the optional GPU dependency is installed.

    Inputs use the project convention [batch, heads, tokens, dim]. FlashAttention expects
    [batch, tokens, heads, dim], so this wrapper handles layout conversion.
    """

    try:
        from flash_attn.flash_attn_interface import flash_attn_func
    except ImportError as exc:
        raise ImportError(
            "FlashAttention is not installed. Install the GPU extra and a compatible "
            "flash-attn build for your CUDA/PyTorch version."
        ) from exc

    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise ValueError("FlashAttention requires CUDA tensors.")
    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()
    output = flash_attn_func(q, k, v, causal=causal)
    return output.transpose(1, 2).contiguous()
