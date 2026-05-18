from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from hogfm.kernels.attention_variants import HyenaOperator, MambaStyleStateSpace

AttentionBackend = Literal[
    "performer",
    "linear",
    "scaled_dot_product",
    "hyena",
    "mamba",
]


@dataclass(frozen=True)
class GenomicTransformerConfig:
    vocab_size: int
    hidden_size: int = 128
    num_layers: int = 4
    num_heads: int = 4
    intermediate_size: int = 512
    max_position_embeddings: int = 65536
    attention_backend: AttentionBackend = "performer"
    performer_features: int = 64
    dropout: float = 0.1
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")


@dataclass
class GenomicModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, length: int, device: torch.device) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.hidden_size, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / self.hidden_size)
        )
        encoding = torch.zeros(length, self.hidden_size, device=device)
        encoding[:, 0::2] = torch.sin(position * div_term)
        if self.hidden_size % 2 == 0:
            encoding[:, 1::2] = torch.cos(position * div_term)
        else:
            encoding[:, 1::2] = torch.cos(position * div_term[:-1])
        return encoding


class PerformerSelfAttention(nn.Module):
    def __init__(self, config: GenomicTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(17)
        projection = torch.randn(
            config.num_heads,
            self.head_dim,
            config.performer_features,
            generator=generator,
        )
        self.register_buffer("projection_matrix", projection, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, tokens, _ = hidden_states.shape
        qkv = self.qkv(hidden_states)
        qkv = qkv.view(batch, tokens, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if self.config.attention_backend == "scaled_dot_product":
            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
            output = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
        else:
            output = self._linear_kernel_attention(query, key, value, attention_mask)

        output = output.transpose(1, 2).contiguous().view(batch, tokens, -1)
        return self.out_proj(self.dropout(output))

    def _linear_kernel_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.config.attention_backend == "performer":
            phi_query = self._favor_feature_map(query)
            phi_key = self._favor_feature_map(key)
        else:
            phi_query = F.elu(query) + 1.0
            phi_key = F.elu(key) + 1.0

        if attention_mask is not None:
            mask = attention_mask[:, None, :, None].to(dtype=phi_key.dtype, device=phi_key.device)
            phi_key = phi_key * mask
            value = value * mask

        kv = torch.einsum("bhnm,bhnd->bhmd", phi_key, value)
        key_sum = phi_key.sum(dim=2)
        denominator = torch.einsum("bhnm,bhm->bhn", phi_query, key_sum).clamp_min(1e-6)
        output = torch.einsum("bhnm,bhmd->bhnd", phi_query, kv)
        return output / denominator.unsqueeze(-1)

    def _favor_feature_map(self, tensor: torch.Tensor) -> torch.Tensor:
        projection = self.projection_matrix.to(device=tensor.device, dtype=tensor.dtype)
        normalized = tensor * (self.head_dim**-0.25)
        projected = torch.einsum("bhnd,hdm->bhnm", normalized, projection)
        squared_norm = normalized.pow(2).sum(dim=-1, keepdim=True) / 2.0
        stabilizer = projected.max(dim=-1, keepdim=True).values.detach()
        features = torch.exp(projected - squared_norm - stabilizer)
        return features / math.sqrt(self.config.performer_features)


class GenomicTransformerBlock(nn.Module):
    def __init__(self, config: GenomicTransformerConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        if config.attention_backend == "hyena":
            self.attention = HyenaOperator(config.hidden_size, config.max_position_embeddings)
        elif config.attention_backend == "mamba":
            self.attention = MambaStyleStateSpace(config.hidden_size)
        else:
            self.attention = PerformerSelfAttention(config)
        self.ffn_norm = nn.LayerNorm(config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(
            self.attention_norm(hidden_states),
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + self.ffn(self.ffn_norm(hidden_states))
        return hidden_states


class GenomicFoundationModel(nn.Module):
    def __init__(self, config: GenomicTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_encoding = SinusoidalPositionEncoding(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([GenomicTransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embeddings.weight

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = enabled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> GenomicModelOutput:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).long()
        positions = self.position_encoding(input_ids.shape[1], input_ids.device)
        hidden_states = self.token_embeddings(input_ids) + positions.unsqueeze(0)
        hidden_states = self.dropout(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return GenomicModelOutput(
            logits=logits,
            loss=loss,
            hidden_states=hidden_states if return_hidden else None,
        )


class GenomicSequenceClassifier(nn.Module):
    def __init__(self, encoder: GenomicFoundationModel, num_labels: int = 2) -> None:
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> GenomicModelOutput:
        output = self.encoder(input_ids, attention_mask=attention_mask, return_hidden=True)
        assert output.hidden_states is not None
        if attention_mask is None:
            attention_mask = input_ids.ne(self.encoder.config.pad_token_id).long()
        mask = attention_mask.unsqueeze(-1).to(dtype=output.hidden_states.dtype)
        pooled = (output.hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        logits = self.classifier(pooled)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return GenomicModelOutput(logits=logits, loss=loss, hidden_states=pooled)


@torch.no_grad()
def sequence_log_likelihood(
    model: GenomicFoundationModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if input_ids.shape[1] < 2:
        return torch.zeros(input_ids.shape[0], device=input_ids.device)
    output = model(input_ids[:, :-1], attention_mask=attention_mask[:, :-1] if attention_mask is not None else None)
    log_probs = F.log_softmax(output.logits, dim=-1)
    targets = input_ids[:, 1:]
    token_scores = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    if attention_mask is not None:
        token_scores = token_scores * attention_mask[:, 1:].to(dtype=token_scores.dtype)
    return token_scores.sum(dim=-1)
