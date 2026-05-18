from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    vocab_size: int = Field(default=134, gt=0)
    hidden_size: int = Field(default=128, gt=0)
    num_layers: int = Field(default=4, gt=0)
    num_heads: int = Field(default=4, gt=0)
    intermediate_size: int = Field(default=512, gt=0)
    max_position_embeddings: int = Field(default=65536, gt=0)
    attention_backend: Literal[
        "performer",
        "linear",
        "scaled_dot_product",
        "hyena",
        "mamba",
    ] = "performer"
    performer_features: int = Field(default=64, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, le=0.9)

    @field_validator("hidden_size")
    @classmethod
    def hidden_size_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("hidden_size must be positive.")
        return value

    def validate_heads(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")


class TokenizerConfig(BaseModel):
    strategy: Literal["nucleotide", "kmer", "bpe"] = "kmer"
    k: int = Field(default=3, ge=1, le=8)
    max_length: int = Field(default=4096, gt=0)


class TrainingConfig(BaseModel):
    seed: int = 13
    batch_size: int = Field(default=8, gt=0)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    epochs: int = Field(default=3, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    gradient_checkpointing: bool = True
    mixed_precision: bool = False
    checkpoint_dir: Path = Path("checkpoints/promoter")


class HogfmConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> HogfmConfig:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        config = cls.model_validate(payload)
        config.model.validate_heads()
        return config
