from __future__ import annotations

import random
from dataclasses import dataclass

DNA_BASES: tuple[str, ...] = ("A", "C", "G", "T")
PROMOTER_MOTIFS: tuple[str, ...] = ("TATAAA", "TATATA", "TATAAT")


@dataclass(frozen=True)
class PromoterExample:
    sequence: str
    label: int
    source: str = "synthetic_promoter_motif"


def _random_dna(length: int, rng: random.Random) -> str:
    return "".join(rng.choice(DNA_BASES) for _ in range(length))


def has_promoter_motif(sequence: str) -> bool:
    upper = sequence.upper()
    return any(motif in upper for motif in PROMOTER_MOTIFS)


def generate_synthetic_promoter_dataset(
    examples: int,
    length: int = 256,
    positive_fraction: float = 0.5,
    seed: int = 13,
) -> list[PromoterExample]:
    if examples <= 0:
        raise ValueError("examples must be positive.")
    if length < max(len(motif) for motif in PROMOTER_MOTIFS):
        raise ValueError("length is too short to hold promoter motifs.")
    rng = random.Random(seed)
    dataset: list[PromoterExample] = []
    positives = int(round(examples * positive_fraction))

    for index in range(examples):
        is_positive = index < positives
        sequence = _random_dna(length, rng)
        if is_positive:
            motif = rng.choice(PROMOTER_MOTIFS)
            insert_at = rng.randint(0, length - len(motif))
            sequence = sequence[:insert_at] + motif + sequence[insert_at + len(motif) :]
        else:
            attempts = 0
            while has_promoter_motif(sequence) and attempts < 100:
                sequence = _random_dna(length, rng)
                attempts += 1
        dataset.append(PromoterExample(sequence=sequence, label=int(is_positive)))

    rng.shuffle(dataset)
    return dataset


def classification_metrics(labels: list[int], probabilities: list[float]) -> dict[str, float]:
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have the same length.")
    if not labels:
        raise ValueError("metrics require at least one example.")

    predictions = [int(prob >= 0.5) for prob in probabilities]
    true_positive = sum(1 for y, pred in zip(labels, predictions) if y == 1 and pred == 1)
    true_negative = sum(1 for y, pred in zip(labels, predictions) if y == 0 and pred == 0)
    false_positive = sum(1 for y, pred in zip(labels, predictions) if y == 0 and pred == 1)
    false_negative = sum(1 for y, pred in zip(labels, predictions) if y == 1 and pred == 0)

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (true_positive + true_negative) / len(labels)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": _binary_auroc(labels, probabilities),
    }


def _binary_auroc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    ranked = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label == 1:
            rank_sum += rank
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
