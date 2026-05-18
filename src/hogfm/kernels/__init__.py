"""Attention kernels and memory estimators."""

from hogfm.kernels.attention_variants import (
    HyenaOperator,
    MambaStyleStateSpace,
    attention_approximation_error,
    chunked_causal_attention,
    rotary_embedding,
)
from hogfm.kernels.flash_attention import flash_attention_available, flash_attention_forward
from hogfm.kernels.linear_attention import estimate_attention_memory, linear_attention
from hogfm.kernels.triton_ops import triton_available, triton_elu_feature_map, triton_rmsnorm

__all__ = [
    "estimate_attention_memory",
    "flash_attention_available",
    "flash_attention_forward",
    "HyenaOperator",
    "linear_attention",
    "MambaStyleStateSpace",
    "attention_approximation_error",
    "chunked_causal_attention",
    "rotary_embedding",
    "triton_available",
    "triton_elu_feature_map",
    "triton_rmsnorm",
]
