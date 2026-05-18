from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


AttentionMode = Literal["linear", "scaled_dot_product"]


@dataclass(frozen=True)
class AttentionMemoryEstimate:
    tokens: int
    heads: int
    dtype_bytes: int
    quadratic_attention_bytes: int
    linear_attention_bytes: int
    reduction_factor: float


def _validate_qkv(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must be shaped [batch, heads, tokens, dim].")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, and value must have identical shapes.")


def elu_feature_map(tensor: torch.Tensor) -> torch.Tensor:
    return F.elu(tensor) + 1.0


def linear_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute causal-free linear attention without forming an N x N score matrix.

    Inputs are expected as [batch, heads, tokens, dim]. The implementation uses the
    standard positive feature-map trick:

        attention(Q, K, V) = phi(Q) @ (phi(K).T @ V) / normalizer

    Memory grows with token count and feature dimension instead of token_count squared.
    """

    _validate_qkv(query, key, value)
    phi_q = elu_feature_map(query)
    phi_k = elu_feature_map(key)

    kv = torch.einsum("bhnd,bhne->bhde", phi_k, value)
    k_sum = phi_k.sum(dim=2)
    normalizer = torch.einsum("bhnd,bhd->bhn", phi_q, k_sum).clamp_min(eps)
    output = torch.einsum("bhnd,bhde->bhne", phi_q, kv)
    return output / normalizer.unsqueeze(-1)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    _validate_qkv(query, key, value)
    return F.scaled_dot_product_attention(query, key, value, is_causal=is_causal)


def attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mode: AttentionMode = "linear",
) -> torch.Tensor:
    if mode == "linear":
        return linear_attention(query, key, value)
    if mode == "scaled_dot_product":
        return scaled_dot_product_attention(query, key, value)
    raise ValueError(f"Unsupported attention mode: {mode}")


def estimate_attention_memory(
    tokens: int,
    heads: int,
    dim: int,
    dtype_bytes: int = 2,
) -> AttentionMemoryEstimate:
    if tokens <= 0 or heads <= 0 or dim <= 0 or dtype_bytes <= 0:
        raise ValueError("tokens, heads, dim, and dtype_bytes must be positive.")

    quadratic = tokens * tokens * heads * dtype_bytes
    linear = tokens * heads * dim * dtype_bytes * 3
    reduction = quadratic / linear if linear else float("inf")
    return AttentionMemoryEstimate(
        tokens=tokens,
        heads=heads,
        dtype_bytes=dtype_bytes,
        quadratic_attention_bytes=quadratic,
        linear_attention_bytes=linear,
        reduction_factor=reduction,
    )
