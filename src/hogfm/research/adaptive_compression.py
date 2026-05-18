from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompressionSpan:
    start: int
    end: int
    token: str
    compression_ratio: float
    reason: str


LOW_COMPLEXITY_BASES: frozenset[str] = frozenset({"A", "C", "G", "T", "N"})


def adaptive_genomic_compression(
    sequence: str,
    min_run: int = 8,
    gc_window: int = 64,
) -> list[CompressionSpan]:
    """Compress low-information genomic regions while preserving boundaries.

    This is the repository's concrete novel systems idea: a deterministic pre-attention
    compression pass that routes repetitive or low-complexity spans into compact summary tokens,
    leaving motif-rich regions at base resolution for the model.
    """

    normalized = "".join(sequence.split()).upper()
    spans: list[CompressionSpan] = []
    index = 0
    while index < len(normalized):
        base = normalized[index]
        if base not in LOW_COMPLEXITY_BASES:
            raise ValueError(f"Unsupported base for compression: {base}")
        run_end = index + 1
        while run_end < len(normalized) and normalized[run_end] == base:
            run_end += 1
        if run_end - index >= min_run:
            spans.append(
                CompressionSpan(
                    start=index,
                    end=run_end,
                    token=f"<run:{base}:{run_end - index}>",
                    compression_ratio=(run_end - index) / 1.0,
                    reason="homopolymer_run",
                )
            )
            index = run_end
            continue
        window_end = min(index + gc_window, len(normalized))
        window = normalized[index:window_end]
        gc = (window.count("G") + window.count("C")) / max(len(window), 1)
        if len(window) == gc_window and (gc < 0.2 or gc > 0.8):
            spans.append(
                CompressionSpan(
                    start=index,
                    end=window_end,
                    token=f"<gc:{gc:.2f}:{len(window)}>",
                    compression_ratio=len(window) / 1.0,
                    reason="extreme_gc_window",
                )
            )
            index = window_end
            continue
        spans.append(
            CompressionSpan(
                start=index,
                end=index + 1,
                token=base,
                compression_ratio=1.0,
                reason="base_resolution",
            )
        )
        index += 1
    return spans


def route_context_windows(
    sequence: str,
    local_window: int = 4096,
    global_stride: int = 65536,
) -> list[tuple[int, int, str]]:
    if local_window <= 0 or global_stride <= 0:
        raise ValueError("local_window and global_stride must be positive.")
    windows: list[tuple[int, int, str]] = []
    for start in range(0, len(sequence), local_window):
        windows.append((start, min(start + local_window, len(sequence)), "local"))
    for start in range(0, len(sequence), global_stride):
        windows.append((start, min(start + local_window, len(sequence)), "global_anchor"))
    return windows
