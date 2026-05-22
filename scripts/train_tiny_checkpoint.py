from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hogfm.checkpointing.manager import build_metadata, export_huggingface_layout, save_checkpoint
from hogfm.models import GenomicFoundationModel, GenomicTransformerConfig
from hogfm.tokenization import GenomicTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export a tiny HOGFM checkpoint.")
    parser.add_argument("--output-dir", default="artifacts/tiny-hogfm")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    tokenizer = GenomicTokenizer(strategy="nucleotide")
    sequences = ["ACGTACGTACGT", "TGCATGCATGCA", "AAAACCCCGGGG", "TTTTGGGGCCCC"]
    encoded = tokenizer.batch_encode(sequences, max_length=16)
    input_ids = torch.tensor(encoded.input_ids, dtype=torch.long)
    labels = input_ids.clone()
    labels[input_ids == tokenizer.pad_token_id] = -100
    model = GenomicFoundationModel(
        GenomicTransformerConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            intermediate_size=64,
            performer_features=16,
            pad_token_id=tokenizer.pad_token_id,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses: list[float] = []
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=input_ids, labels=labels)
        assert output.loss is not None
        output.loss.backward()
        optimizer.step()
        losses.append(float(output.loss.detach()))

    output_dir = Path(args.output_dir)
    metadata = build_metadata(
        step=args.steps,
        epoch=1,
        metrics={"loss": losses[-1]},
        config=model.config.__dict__,
    )
    save_checkpoint(model, output_dir, metadata, optimizer=optimizer, use_safetensors=True)
    export_huggingface_layout(model, tokenizer, output_dir / "hf_export", model.config.__dict__)
    (output_dir / "training_log.json").write_text(
        json.dumps({"losses": losses, "seed": args.seed}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "losses": losses}, indent=2))


if __name__ == "__main__":
    main()
