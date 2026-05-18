from __future__ import annotations

from dataclasses import dataclass

DNA_ALPHABET: frozenset[str] = frozenset({"A", "C", "G", "T", "N"})
TOKEN_TO_ID: dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


@dataclass(frozen=True)
class FastaRecord:
    """A validated FASTA sequence record."""

    name: str
    sequence: str

    @property
    def length(self) -> int:
        return len(self.sequence)


@dataclass(frozen=True)
class VariantContext:
    """Sequence window surrounding a single-nucleotide or short indel variant."""

    contig: str
    position: int
    reference: str
    alternate: str
    start: int
    end: int
    reference_context: str
    alternate_context: str
    token_ids: list[int]


def normalize_sequence(sequence: str) -> str:
    normalized = "".join(sequence.split()).upper()
    invalid = sorted(set(normalized) - DNA_ALPHABET)
    if invalid:
        joined = ", ".join(invalid)
        raise ValueError(f"Unsupported DNA symbols found: {joined}")
    return normalized


def parse_fasta(text: str) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    name: str | None = None
    chunks: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records.append(FastaRecord(name=name, sequence=normalize_sequence("".join(chunks))))
            name = line[1:].strip() or f"record_{len(records) + 1}"
            chunks = []
        else:
            if name is None:
                raise ValueError("FASTA sequence data appeared before a header line.")
            chunks.append(line)

    if name is not None:
        records.append(FastaRecord(name=name, sequence=normalize_sequence("".join(chunks))))

    if not records:
        raise ValueError("No FASTA records were found.")
    return records


def tokenize_dna(sequence: str) -> list[int]:
    return [TOKEN_TO_ID[base] for base in normalize_sequence(sequence)]


def build_variant_context(
    record: FastaRecord,
    position: int,
    reference: str,
    alternate: str,
    flank: int = 512,
) -> VariantContext:
    if position < 1:
        raise ValueError("Variant position must be 1-based and positive.")
    if flank < 0:
        raise ValueError("Flank must be non-negative.")

    ref = normalize_sequence(reference)
    alt = normalize_sequence(alternate)
    zero_based = position - 1
    ref_end = zero_based + len(ref)

    if ref_end > record.length:
        raise ValueError("Variant reference extends beyond the sequence length.")

    observed_ref = record.sequence[zero_based:ref_end]
    if observed_ref != ref:
        raise ValueError(
            f"Reference allele mismatch at {record.name}:{position}; "
            f"expected {ref}, observed {observed_ref}."
        )

    start = max(0, zero_based - flank)
    end = min(record.length, ref_end + flank)
    prefix = record.sequence[start:zero_based]
    suffix = record.sequence[ref_end:end]
    reference_context = prefix + ref + suffix
    alternate_context = prefix + alt + suffix

    return VariantContext(
        contig=record.name,
        position=position,
        reference=ref,
        alternate=alt,
        start=start + 1,
        end=end,
        reference_context=reference_context,
        alternate_context=alternate_context,
        token_ids=tokenize_dna(alternate_context),
    )


def parse_vcf_line(line: str) -> tuple[str, int, str, str]:
    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError("VCF-like rows must contain CHROM POS ID REF ALT at minimum.")

    contig = parts[0]
    try:
        position = int(parts[1])
    except ValueError as exc:
        raise ValueError("VCF position must be an integer.") from exc

    reference = normalize_sequence(parts[3])
    alternate = normalize_sequence(parts[4].split(",")[0])
    return contig, position, reference, alternate
