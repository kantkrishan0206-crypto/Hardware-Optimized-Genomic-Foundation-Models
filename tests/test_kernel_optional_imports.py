import pytest


def test_optional_kernel_availability_flags() -> None:
    pytest.importorskip("torch")
    from hogfm.kernels import flash_attention_available, triton_available

    assert isinstance(flash_attention_available(), bool)
    assert isinstance(triton_available(), bool)
