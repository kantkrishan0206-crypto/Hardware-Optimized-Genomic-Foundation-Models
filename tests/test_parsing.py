from hogfm.parsing.fasta import build_variant_context, parse_fasta, parse_vcf_line, tokenize_dna


def test_parse_fasta_records() -> None:
    records = parse_fasta(">chr1\nACGTN\n>chr2\nGGCC\n")
    assert len(records) == 2
    assert records[0].name == "chr1"
    assert records[0].sequence == "ACGTN"
    assert records[1].length == 4


def test_variant_context_replaces_reference() -> None:
    record = parse_fasta(">chr1\nAACCGGTT\n")[0]
    context = build_variant_context(record, position=3, reference="C", alternate="T", flank=2)
    assert context.reference_context == "AACCG"
    assert context.alternate_context == "AATCG"
    assert context.token_ids == tokenize_dna("AATCG")


def test_parse_vcf_line() -> None:
    contig, position, reference, alternate = parse_vcf_line("chr1 42 . A G . PASS .")
    assert contig == "chr1"
    assert position == 42
    assert reference == "A"
    assert alternate == "G"
