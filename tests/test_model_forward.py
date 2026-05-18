import pytest


def test_model_forward_backward() -> None:
    torch = pytest.importorskip("torch")
    from hogfm.models import GenomicFoundationModel, GenomicTransformerConfig
    from hogfm.tokenization import GenomicTokenizer

    tokenizer = GenomicTokenizer(strategy="nucleotide")
    batch = tokenizer.batch_encode(["ACGTACGT", "TGCATGCA"], max_length=12)
    input_ids = torch.tensor(batch.input_ids, dtype=torch.long)
    labels = input_ids.clone()
    labels[input_ids == tokenizer.pad_token_id] = -100
    model = GenomicFoundationModel(
        GenomicTransformerConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            intermediate_size=64,
            performer_features=16,
        )
    )
    output = model(input_ids=input_ids, labels=labels)
    assert output.loss is not None
    output.loss.backward()
    assert any(param.grad is not None for param in model.parameters())
