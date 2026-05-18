import pytest


def test_linear_attention_shape() -> None:
    torch = pytest.importorskip("torch")
    from hogfm.kernels.linear_attention import linear_attention

    query = torch.randn(2, 3, 16, 8)
    key = torch.randn(2, 3, 16, 8)
    value = torch.randn(2, 3, 16, 8)
    output = linear_attention(query, key, value)
    assert output.shape == value.shape
    assert torch.isfinite(output).all()


def test_memory_estimate_reduces_quadratic_cost() -> None:
    pytest.importorskip("torch")
    from hogfm.kernels.linear_attention import estimate_attention_memory

    estimate = estimate_attention_memory(tokens=1024, heads=8, dim=64, dtype_bytes=2)
    assert estimate.quadratic_attention_bytes > estimate.linear_attention_bytes
    assert estimate.reduction_factor > 1.0
