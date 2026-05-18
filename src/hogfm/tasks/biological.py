from __future__ import annotations

from dataclasses import dataclass

from hogfm.tasks.promoter import classification_metrics


@dataclass(frozen=True)
class GenomicTaskExample:
    sequence: str
    label: int
    task: str
    metadata: dict[str, str]


ENHANCER_MOTIFS: tuple[str, ...] = ("CACGTG", "GGAA", "TGACGT")
SPLICE_DONOR: tuple[str, ...] = ("GT", "GC")
SPLICE_ACCEPTOR: tuple[str, ...] = ("AG",)
BRCA1_HOTSPOT_WINDOWS: tuple[str, ...] = ("CAGT", "AGAA", "TCTG")


def enhancer_label(sequence: str) -> int:
    upper = sequence.upper()
    return int(any(motif in upper for motif in ENHANCER_MOTIFS))


def splice_site_label(sequence: str) -> int:
    upper = sequence.upper()
    if len(upper) < 4:
        return 0
    center = len(upper) // 2
    donor = upper[center : center + 2]
    acceptor = upper[max(center - 2, 0) : center]
    return int(donor in SPLICE_DONOR or acceptor in SPLICE_ACCEPTOR)


def brca1_variant_prior(reference: str, alternate: str) -> float:
    ref = reference.upper()
    alt = alternate.upper()
    hotspot_shift = sum(motif in alt for motif in BRCA1_HOTSPOT_WINDOWS)
    disruptive = int(len(ref) != len(alt) or alt.count("N") > ref.count("N"))
    gc_delta = abs((alt.count("G") + alt.count("C")) - (ref.count("G") + ref.count("C")))
    return min(1.0, 0.15 + 0.2 * hotspot_shift + 0.35 * disruptive + 0.03 * gc_delta)


def clinvar_class_label(significance: str) -> int:
    normalized = significance.lower()
    pathogenic_terms = ("pathogenic", "likely_pathogenic")
    benign_terms = ("benign", "likely_benign")
    if any(term in normalized for term in pathogenic_terms):
        return 1
    if any(term in normalized for term in benign_terms):
        return 0
    raise ValueError(f"Unsupported ClinVar significance label: {significance}")


def confusion_matrix(labels: list[int], probabilities: list[float]) -> dict[str, int]:
    predictions = [int(probability >= 0.5) for probability in probabilities]
    return {
        "true_positive": sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 1),
        "true_negative": sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 0),
        "false_positive": sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 1),
        "false_negative": sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 0),
    }


def evaluate_binary_task(labels: list[int], probabilities: list[float]) -> dict[str, float | int]:
    metrics = classification_metrics(labels, probabilities)
    metrics.update(confusion_matrix(labels, probabilities))
    return metrics
