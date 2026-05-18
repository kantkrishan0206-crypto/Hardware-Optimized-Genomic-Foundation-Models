from hogfm.benchmarks import compare_attention_families


def test_compare_attention_families() -> None:
    rows = compare_attention_families([128], heads=2, dim=16)
    families = {row.model_family for row in rows}
    assert "vanilla_transformer" in families
    assert "hyenadna" in families
