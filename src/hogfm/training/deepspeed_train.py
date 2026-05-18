from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from hogfm.models.peft_adapter import LoraAdapterConfig, apply_lora_adapter, save_adapter


@dataclass(frozen=True)
class TrainingRunConfig:
    base_model: str
    dataset_path: Path
    output_dir: Path
    deepspeed_config: Path
    max_steps: int
    learning_rate: float


def _import_training_stack() -> tuple[Any, Any, Any, Any]:
    try:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise ImportError(
            "Training requires datasets and transformers. Install the optional GPU stack and datasets: "
            'pip install "hardware-optimized-genomic-foundation-models[gpu]" datasets'
        ) from exc
    return load_dataset, AutoModelForCausalLM, AutoTokenizer, (Trainer, TrainingArguments)


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Model config YAML must contain a mapping at the top level.")
    return payload


def build_run_config(args: argparse.Namespace) -> TrainingRunConfig:
    config = load_yaml_config(Path(args.model_config))
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    return TrainingRunConfig(
        base_model=args.base_model or model_cfg.get("base_model"),
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        deepspeed_config=Path(args.deepspeed_config),
        max_steps=int(args.max_steps or training_cfg.get("max_steps", 1000)),
        learning_rate=float(args.learning_rate or training_cfg.get("learning_rate", 2e-4)),
    )


def train(config: TrainingRunConfig, lora_config: LoraAdapterConfig) -> Path:
    load_dataset, AutoModelForCausalLM, AutoTokenizer, trainer_types = _import_training_stack()
    Trainer, TrainingArguments = trainer_types

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=str(config.dataset_path), split="train")

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(batch["sequence"], truncation=True, padding="max_length")

    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=dataset.column_names)
    model = AutoModelForCausalLM.from_pretrained(config.base_model, trust_remote_code=True)
    model = apply_lora_adapter(model, lora_config)

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        deepspeed=str(config.deepspeed_config),
        report_to=[],
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
    trainer.train()
    return save_adapter(model, config.output_dir / "adapter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSpeed ZeRO-3 PEFT training entry point.")
    parser.add_argument("--dataset", required=True, help="JSONL file with a `sequence` field.")
    parser.add_argument("--output-dir", default="outputs/hogfm-run")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--deepspeed-config", default="configs/deepspeed_zero3.json")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_config = build_run_config(args)
    lora_config = LoraAdapterConfig()
    adapter_path = train(run_config, lora_config)
    print(f"Saved adapter to {adapter_path}")


if __name__ == "__main__":
    main()
