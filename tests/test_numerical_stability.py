import pytest


def test_linear_attention_numerical_stability() -> None:
    torch = pytest.importorskip("torch")
    from hogfm.kernels.linear_attention import linear_attention

    query = torch.randn(1, 2, 32, 16) * 0.01
    key = torch.randn(1, 2, 32, 16) * 0.01
    value = torch.randn(1, 2, 32, 16)
    output = linear_attention(query, key, value)
    assert torch.isfinite(output).all()


def test_deterministic_model_initialization() -> None:
    torch = pytest.importorskip("torch")
    from hogfm.models import GenomicFoundationModel, GenomicTransformerConfig

    torch.manual_seed(123)
    first = GenomicFoundationModel(GenomicTransformerConfig(vocab_size=16, hidden_size=32))
    torch.manual_seed(123)
    second = GenomicFoundationModel(GenomicTransformerConfig(vocab_size=16, hidden_size=32))
    first_param = next(first.parameters())
    second_param = next(second.parameters())
    assert torch.allclose(first_param, second_param)
