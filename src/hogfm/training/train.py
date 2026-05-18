from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from hogfm.models import GenomicFoundationModel, GenomicSequenceClassifier, GenomicTransformerConfig
from hogfm.observability import ExperimentTracker
from hogfm.tasks.promoter import classification_metrics
from hogfm.tokenization import GenomicTokenizer


class JsonlSequenceDataset(Dataset[dict[str, Any]]):
    def __init__(self, path: str | Path) -> None:
        self.rows: list[dict[str, Any]] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                self.rows.append({"sequence": payload["sequence"], "label": int(payload["label"])})
        if not self.rows:
            raise ValueError(f"No examples found in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def collate_batch(
    rows: list[dict[str, Any]],
    tokenizer: GenomicTokenizer,
    max_length: int,
) -> dict[str, torch.Tensor]:
    encoding = tokenizer.batch_encode([row["sequence"] for row in rows], max_length=max_length)
    labels = [row["label"] for row in rows]
    return {
        "input_ids": torch.tensor(encoding.input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(encoding.attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def evaluate(
    model: GenomicSequenceClassifier,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(output.logits, dim=-1)[:, 1]
            labels.extend(batch_labels.cpu().tolist())
            probabilities.extend(probs.cpu().tolist())
    return classification_metrics(labels, probabilities)


def train_classifier(args: argparse.Namespace) -> dict[str, float]:
    set_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = GenomicTokenizer(strategy=args.tokenizer, k=args.k)
    train_dataset = JsonlSequenceDataset(args.train)
    valid_dataset = JsonlSequenceDataset(args.valid)
    collate = lambda rows: collate_batch(rows, tokenizer, args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    config = GenomicTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        num_heads=args.heads,
        intermediate_size=args.hidden_size * 4,
        attention_backend=args.attention,
        performer_features=args.performer_features,
        pad_token_id=tokenizer.pad_token_id,
    )
    encoder = GenomicFoundationModel(config)
    model = GenomicSequenceClassifier(encoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    tracker = ExperimentTracker(args.output_dir, run_name="promoter-classifier")

    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            assert output.loss is not None
            output.loss.backward()
            if step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        metrics = evaluate(model, valid_loader, device)
        tracker.log_metrics(metrics, step=epoch + 1)
        print(f"epoch={epoch + 1} metrics={json.dumps(metrics, sort_keys=True)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config.__dict__},
        output_dir / "promoter_classifier.pt",
    )
    tokenizer.save_pretrained(output_dir)
    metrics = evaluate(model, valid_loader, device)
    tracker.log_metrics(metrics, step=args.epochs + 1)
    tracker.close()
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train promoter classifier with HOGFM encoder.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--output-dir", default="checkpoints/promoter")
    parser.add_argument("--tokenizer", choices=["nucleotide", "kmer"], default="kmer")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--performer-features", type=int, default=64)
    parser.add_argument("--attention", choices=["performer", "linear"], default="performer")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    metrics = train_classifier(parse_args())
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
