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

    @triton.jit
    def _rmsnorm_kernel(x_ptr, w_ptr, y_ptr, rows, cols, eps, block_size: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_size)
        mask = offsets < cols
        values = tl.load(x_ptr + row * cols + offsets, mask=mask, other=0.0)
        weights = tl.load(w_ptr + offsets, mask=mask, other=1.0)
        variance = tl.sum(values * values, axis=0) / cols
        inv_rms = tl.rsqrt(variance + eps)
        tl.store(y_ptr + row * cols + offsets, values * inv_rms * weights, mask=mask)

else:
    _elu_plus_one_kernel = None
    _rmsnorm_kernel = None


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


def triton_rmsnorm(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    block_size: int = 1024,
) -> torch.Tensor:
    if _rmsnorm_kernel is None or triton is None:
        raise ImportError("Triton is not installed. Install the optional GPU stack first.")
    if not tensor.is_cuda or not weight.is_cuda:
        raise ValueError("The Triton RMSNorm kernel requires CUDA tensors.")
    if tensor.shape[-1] > block_size:
        raise ValueError("Increase block_size for this hidden dimension.")
    contiguous = tensor.contiguous()
    output = torch.empty_like(contiguous)
    rows = contiguous.numel() // contiguous.shape[-1]
    _rmsnorm_kernel[(rows,)](
        contiguous,
        weight.contiguous(),
        output,
        rows,
        contiguous.shape[-1],
        eps,
        block_size,
    )
    return output.view_as(tensor)
