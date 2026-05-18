from hogfm.research import adaptive_genomic_compression, route_context_windows


def test_adaptive_compression_collapses_homopolymer() -> None:
    spans = adaptive_genomic_compression("AAAAAAAACGT", min_run=4)
    assert spans[0].token == "<run:A:8>"
    assert spans[0].compression_ratio == 8.0


def test_context_routing_adds_global_anchors() -> None:
    windows = route_context_windows("A" * 100, local_window=32, global_stride=64)
    labels = {window[2] for window in windows}
    assert labels == {"local", "global_anchor"}
