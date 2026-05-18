from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def chunked_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Exact causal attention computed in chunks to reduce peak score memory."""

    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, and value must have identical shapes.")
    if query.ndim != 4:
        raise ValueError("Expected tensors shaped [batch, heads, tokens, dim].")
    outputs: list[torch.Tensor] = []
    scale = query.shape[-1] ** -0.5
    tokens = query.shape[2]
    for start in range(0, tokens, chunk_size):
        end = min(start + chunk_size, tokens)
        q_chunk = query[:, :, start:end, :]
        scores = torch.einsum("bhqd,bhkd->bhqk", q_chunk, key) * scale
        key_positions = torch.arange(tokens, device=query.device)
        query_positions = torch.arange(start, end, device=query.device)
        causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        scores = scores.masked_fill(~causal_mask[None, None, :, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("bhqk,bhkd->bhqd", weights, value))
    return torch.cat(outputs, dim=2)


class HyenaOperator(nn.Module):
    """Compact Hyena-style implicit long convolution operator.

    This is a production-safe reference implementation, not the fused HazyResearch kernel.
    It provides the same modeling role for correctness tests and CPU experiments.
    """

    def __init__(self, hidden_size: int, max_filter_length: int = 4096) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.max_filter_length = max_filter_length
        self.in_proj = nn.Linear(hidden_size, hidden_size * 2)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.filter_bias = nn.Parameter(torch.zeros(hidden_size))
        self.filter_decay = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values, gates = self.in_proj(hidden_states).chunk(2, dim=-1)
        if attention_mask is not None:
            values = values * attention_mask.unsqueeze(-1).to(values.dtype)
        batch, tokens, channels = values.shape
        length = min(tokens, self.max_filter_length)
        positions = torch.arange(length, device=values.device, dtype=values.dtype)
        decay = F.softplus(self.filter_decay).view(1, channels)
        kernel = torch.exp(-positions[:, None] / decay) + self.filter_bias.view(1, channels)
        padded = F.pad(values.transpose(1, 2), (length - 1, 0))
        filters = kernel.flip(0).transpose(0, 1).unsqueeze(1)
        convolved = F.conv1d(padded, filters, groups=channels)
        convolved = convolved[:, :, :tokens].transpose(1, 2)
        return self.out_proj(convolved * torch.sigmoid(gates))


class MambaStyleStateSpace(nn.Module):
    """Reference selective state-space mixer inspired by Mamba."""

    def __init__(self, hidden_size: int, state_size: int = 16) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.in_proj = nn.Linear(hidden_size, hidden_size * 2)
        self.delta_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.a_log = nn.Parameter(torch.zeros(hidden_size, state_size))
        self.b_proj = nn.Linear(hidden_size, hidden_size * state_size)
        self.c_proj = nn.Linear(hidden_size, hidden_size * state_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values, gates = self.in_proj(hidden_states).chunk(2, dim=-1)
        delta = F.softplus(self.delta_proj(hidden_states))
        a = -torch.exp(self.a_log)
        b = self.b_proj(hidden_states).view(*hidden_states.shape[:2], self.hidden_size, self.state_size)
        c = self.c_proj(hidden_states).view(*hidden_states.shape[:2], self.hidden_size, self.state_size)
        state = torch.zeros(
            hidden_states.shape[0],
            self.hidden_size,
            self.state_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        outputs: list[torch.Tensor] = []
        for index in range(hidden_states.shape[1]):
            delta_t = delta[:, index, :, None]
            state = torch.exp(delta_t * a) * state + b[:, index] * values[:, index, :, None]
            out = (state * c[:, index]).sum(dim=-1)
            if attention_mask is not None:
                out = out * attention_mask[:, index].unsqueeze(-1).to(out.dtype)
            outputs.append(out)
        stacked = torch.stack(outputs, dim=1)
        return self.out_proj(stacked * torch.sigmoid(gates))


def rotary_embedding(
    query: torch.Tensor,
    key: torch.Tensor,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query.shape != key.shape:
        raise ValueError("query and key must have the same shape.")
    dim = query.shape[-1]
    half = dim // 2
    positions = torch.arange(query.shape[-2], device=query.device, dtype=query.dtype)
    inv_freq = base ** (-torch.arange(0, half, device=query.device, dtype=query.dtype) / half)
    angles = torch.einsum("n,d->nd", positions, inv_freq)
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]
    return _apply_rotary(query, cos, sin), _apply_rotary(key, cos, sin)


def _apply_rotary(tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    first = tensor[..., : cos.shape[-1]]
    second = tensor[..., cos.shape[-1] : cos.shape[-1] * 2]
    rest = tensor[..., cos.shape[-1] * 2 :]
    rotated = torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)
    if rest.numel() == 0:
        return rotated
    return torch.cat((rotated, rest), dim=-1)


def attention_approximation_error(
    approximate: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float]:
    delta = approximate.detach() - reference.detach()
    return {
        "max_abs_error": delta.abs().max().item(),
        "mean_abs_error": delta.abs().mean().item(),
        "rmse": math.sqrt(delta.pow(2).mean().item()),
    }
