# Dataset Pipeline Guide

## Supported Formats

| Format | Reader |
| --- | --- |
| FASTA | `hogfm.data.stream_fasta_records` |
| FASTQ | `hogfm.data.stream_fastq` |
| VCF | `hogfm.data.stream_vcf` |
| BED | `hogfm.data.stream_bed` |
| GTF | `hogfm.data.stream_gtf` |

## Reference Datasets

| Dataset | Purpose | Notes |
| --- | --- | --- |
| HG38 | reference sequence context | pin genome build and checksum |
| ClinVar | pathogenicity labels | normalize significance strings |
| ENCODE | regulatory elements | promoters/enhancers/TF binding |
| GTEx | tissue expression | expression-linked genomic regions |
| GenomicBenchmarks | downstream tasks | benchmark-friendly splits |

## Sharding

Use `hogfm.data.shard_records(records, shard_index, shard_count)` to partition streams across
distributed preprocessing workers.

## Validation Checklist

- record source URL
- genome build
- checksum
- preprocessing command
- train/valid/test split seed
- label distribution
- leakage checks across genomic intervals
