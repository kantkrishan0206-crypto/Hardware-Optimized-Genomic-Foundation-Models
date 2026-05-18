from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def triton_available() -> bool:
    return triton is not None


if triton is not None and tl is not None:

    @triton.jit
    def _elu_plus_one_kernel(x_ptr, y_ptr, n_elements, block_size: tl.constexpr):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements
        values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        exp_values = tl.exp(values)
        output = tl.where(values > 0.0, values + 1.0, exp_values)
        tl.store(y_ptr + offsets, output, mask=mask)

else:
    _elu_plus_one_kernel = None


def triton_elu_feature_map(tensor: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    """Apply the linear-attention ELU+1 feature map with a fused Triton kernel."""

    if _elu_plus_one_kernel is None or triton is None:
        raise ImportError("Triton is not installed. Install the optional GPU stack first.")
    if not tensor.is_cuda:
        raise ValueError("The Triton feature-map kernel requires a CUDA tensor.")
    contiguous = tensor.contiguous()
    output = torch.empty_like(contiguous)
    grid = (triton.cdiv(contiguous.numel(), block_size),)
    _elu_plus_one_kernel[grid](contiguous, output, contiguous.numel(), block_size)
    return output
