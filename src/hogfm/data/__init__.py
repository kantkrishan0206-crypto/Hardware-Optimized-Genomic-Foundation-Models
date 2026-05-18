"""Streaming genomic dataset ingestion."""

from hogfm.data.readers import (
    BedRecord,
    FastqRecord,
    GtfRecord,
    VcfRecord,
    shard_records,
    stream_bed,
    stream_fasta_records,
    stream_fastq,
    stream_gtf,
    stream_vcf,
)

__all__ = [
    "BedRecord",
    "FastqRecord",
    "GtfRecord",
    "VcfRecord",
    "shard_records",
    "stream_bed",
    "stream_fasta_records",
    "stream_fastq",
    "stream_gtf",
    "stream_vcf",
]
