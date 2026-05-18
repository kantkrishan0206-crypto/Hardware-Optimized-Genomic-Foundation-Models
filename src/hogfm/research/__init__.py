"""Novel research components for genomic sequence systems."""

from hogfm.research.adaptive_compression import (
    CompressionSpan,
    adaptive_genomic_compression,
    route_context_windows,
)

__all__ = ["CompressionSpan", "adaptive_genomic_compression", "route_context_windows"]
