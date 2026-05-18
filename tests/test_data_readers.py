from pathlib import Path

from hogfm.data import stream_bed, stream_fasta_records, stream_fastq, stream_vcf


def test_stream_fasta_records(tmp_path: Path) -> None:
    path = tmp_path / "sample.fa"
    path.write_text(">chr1\nACGT\n", encoding="utf-8")
    records = list(stream_fasta_records(path))
    assert records[0].name == "chr1"
    assert records[0].sequence == "ACGT"


def test_stream_fastq(tmp_path: Path) -> None:
    path = tmp_path / "sample.fq"
    path.write_text("@read1\nACGT\n+\n!!!!\n", encoding="utf-8")
    records = list(stream_fastq(path))
    assert records[0].quality == "!!!!"


def test_stream_vcf_and_bed(tmp_path: Path) -> None:
    vcf = tmp_path / "sample.vcf"
    vcf.write_text("#header\nchr1\t2\t.\tA\tG\t.\tPASS\t.\n", encoding="utf-8")
    bed = tmp_path / "sample.bed"
    bed.write_text("chr1\t0\t10\tpeak\n", encoding="utf-8")
    assert list(stream_vcf(vcf))[0].alternate == "G"
    assert list(stream_bed(bed))[0].name == "peak"
