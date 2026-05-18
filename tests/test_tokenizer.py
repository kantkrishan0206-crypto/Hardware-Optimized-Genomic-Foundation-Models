from hogfm.tokenization import GenomicTokenizer, train_bpe_tokenizer


def test_kmer_tokenizer_round_trip() -> None:
    tokenizer = GenomicTokenizer(strategy="kmer", k=3)
    ids = tokenizer.encode("ACGTAC", add_special_tokens=False)
    assert len(ids) == 4
    assert all(isinstance(item, int) for item in ids)


def test_batch_padding() -> None:
    tokenizer = GenomicTokenizer(strategy="nucleotide")
    batch = tokenizer.batch_encode(["ACG", "ACGT"], max_length=8)
    assert len(batch.input_ids) == 2
    assert batch.attention_mask[0][-1] == 0
    assert batch.attention_mask[1][0] == 1


def test_bpe_training_adds_merges() -> None:
    tokenizer = train_bpe_tokenizer(["ATATAT", "ATATGC", "ATATAT"], vocab_size=16)
    assert tokenizer.strategy == "bpe"
    assert tokenizer.merges
