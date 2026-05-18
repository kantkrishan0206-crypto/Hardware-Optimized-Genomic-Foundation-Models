from __future__ import annotations

import gzip
import mmap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, TextIO, TypeVar

from hogfm.parsing.fasta import FastaRecord, normalize_sequence


@dataclass(frozen=True)
class FastqRecord:
    name: str
    sequence: str
    quality: str


@dataclass(frozen=True)
class VcfRecord:
    contig: str
    position: int
    identifier: str
    reference: str
    alternate: str
    quality: str
    filter_status: str
    info: str


@dataclass(frozen=True)
class BedRecord:
    contig: str
    start: int
    end: int
    name: str | None = None


@dataclass(frozen=True)
class GtfRecord:
    contig: str
    source: str
    feature: str
    start: int
    end: int
    score: str
    strand: str
    frame: str
    attributes: dict[str, str]


T = TypeVar("T")


def _open_text(path: str | Path) -> TextIO:
    target = Path(path)
    if target.suffix == ".gz":
        return gzip.open(target, "rt", encoding="utf-8")
    return target.open("r", encoding="utf-8")


def stream_fasta_records(path: str | Path) -> Iterator[FastaRecord]:
    name: str | None = None
    chunks: list[str] = []
    with _open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield FastaRecord(name=name, sequence=normalize_sequence("".join(chunks)))
                name = line[1:].strip() or "unnamed"
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        yield FastaRecord(name=name, sequence=normalize_sequence("".join(chunks)))


def stream_fastq(path: str | Path) -> Iterator[FastqRecord]:
    with _open_text(path) as handle:
        while True:
            header = handle.readline().rstrip()
            if not header:
                break
            sequence = handle.readline().rstrip()
            plus = handle.readline().rstrip()
            quality = handle.readline().rstrip()
            if not header.startswith("@") or plus != "+":
                raise ValueError("Malformed FASTQ record.")
            if len(sequence) != len(quality):
                raise ValueError("FASTQ sequence and quality lengths differ.")
            yield FastqRecord(header[1:], normalize_sequence(sequence), quality)


def stream_vcf(path: str | Path) -> Iterator[VcfRecord]:
    with _open_text(path) as handle:
        for raw_line in handle:
            if raw_line.startswith("#") or not raw_line.strip():
                continue
            fields = raw_line.rstrip().split("\t")
            if len(fields) < 8:
                raise ValueError("VCF records require at least 8 columns.")
            yield VcfRecord(
                contig=fields[0],
                position=int(fields[1]),
                identifier=fields[2],
                reference=normalize_sequence(fields[3]),
                alternate=normalize_sequence(fields[4].split(",")[0]),
                quality=fields[5],
                filter_status=fields[6],
                info=fields[7],
            )


def stream_bed(path: str | Path) -> Iterator[BedRecord]:
    with _open_text(path) as handle:
        for raw_line in handle:
            if raw_line.startswith("#") or not raw_line.strip():
                continue
            fields = raw_line.rstrip().split("\t")
            if len(fields) < 3:
                raise ValueError("BED records require at least 3 columns.")
            yield BedRecord(
                contig=fields[0],
                start=int(fields[1]),
                end=int(fields[2]),
                name=fields[3] if len(fields) > 3 else None,
            )


def stream_gtf(path: str | Path) -> Iterator[GtfRecord]:
    with _open_text(path) as handle:
        for raw_line in handle:
            if raw_line.startswith("#") or not raw_line.strip():
                continue
            fields = raw_line.rstrip().split("\t")
            if len(fields) != 9:
                raise ValueError("GTF records require exactly 9 columns.")
            yield GtfRecord(
                contig=fields[0],
                source=fields[1],
                feature=fields[2],
                start=int(fields[3]),
                end=int(fields[4]),
                score=fields[5],
                strand=fields[6],
                frame=fields[7],
                attributes=_parse_gtf_attributes(fields[8]),
            )


def _parse_gtf_attributes(raw: str) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for item in raw.strip().rstrip(";").split(";"):
        item = item.strip()
        if not item:
            continue
        key, _, value = item.partition(" ")
        attributes[key] = value.strip().strip('"')
    return attributes


def shard_records(records: Iterable[T], shard_index: int, shard_count: int) -> Iterator[T]:
    if shard_count <= 0 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError("Invalid shard_index/shard_count combination.")
    for index, record in enumerate(records):
        if index % shard_count == shard_index:
            yield record


def mmap_text(path: str | Path) -> mmap.mmap:
    handle = Path(path).open("r+b")
    return mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
