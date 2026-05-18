from hogfm.tasks import (
    brca1_variant_prior,
    clinvar_class_label,
    confusion_matrix,
    enhancer_label,
    splice_site_label,
)


def test_biological_task_labels() -> None:
    assert enhancer_label("AAAACACGTGAAAA") == 1
    assert splice_site_label("AAAAAGTAAAA") == 1
    assert clinvar_class_label("Pathogenic") == 1
    assert brca1_variant_prior("ACGT", "ACGTT") > 0.1


def test_confusion_matrix() -> None:
    matrix = confusion_matrix([0, 1, 1, 0], [0.1, 0.8, 0.9, 0.7])
    assert matrix["true_positive"] == 2
    assert matrix["false_positive"] == 1
