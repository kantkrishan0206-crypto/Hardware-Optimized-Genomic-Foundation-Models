import pytest


def test_chunked_causal_attention_shape_and_grad() -> None:
    torch = pytest.importorskip("torch")
    from hogfm.kernels import chunked_causal_attention

    query = torch.randn(1, 2, 8, 4, requires_grad=True)
    key = torch.randn(1, 2, 8, 4, requires_grad=True)
    value = torch.randn(1, 2, 8, 4, requires_grad=True)
    output = chunked_causal_attention(query, key, value, chunk_size=4)
    assert output.shape == value.shape
    loss = output.pow(2).mean()
    loss.backward()
    assert query.grad is not None


def test_hyena_and_mamba_mixers() -> None:
    torch = pytest.importorskip("torch")
    from hogfm.kernels import HyenaOperator, MambaStyleStateSpace

    hidden = torch.randn(2, 8, 16)
    mask = torch.ones(2, 8, dtype=torch.long)
    assert HyenaOperator(16)(hidden, mask).shape == hidden.shape
    assert MambaStyleStateSpace(16, state_size=4)(hidden, mask).shape == hidden.shape
