import pytest


@pytest.mark.gpu
def test_cuda_torch_available() -> None:
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available()


@pytest.mark.gpu
def test_triton_kernel_contract() -> None:
    torch = pytest.importorskip("torch")
    triton = pytest.importorskip("triton")
    from hogfm.kernels import triton_elu_feature_map

    tensor = torch.randn(128, device="cuda")
    output = triton_elu_feature_map(tensor)
    assert output.is_cuda
    assert triton is not None
