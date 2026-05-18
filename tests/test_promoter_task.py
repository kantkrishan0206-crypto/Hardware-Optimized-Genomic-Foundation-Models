from hogfm.tasks.promoter import classification_metrics, generate_synthetic_promoter_dataset


def test_synthetic_promoter_dataset_has_labels() -> None:
    dataset = generate_synthetic_promoter_dataset(examples=20, length=64, seed=7)
    labels = {example.label for example in dataset}
    assert labels == {0, 1}


def test_classification_metrics() -> None:
    metrics = classification_metrics([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2])
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["auroc"] == 1.0
