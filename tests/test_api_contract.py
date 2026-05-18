import pytest


def test_api_openapi_contract() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("fastapi")
    from hogfm.api.main import app

    schema = app.openapi()
    paths = schema["paths"]
    assert "/predict" in paths
    assert "/score_variant" in paths
    assert "/embed_sequence" in paths
    assert "/generate" in paths
    assert "/benchmark" in paths
