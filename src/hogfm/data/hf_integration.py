from __future__ import annotations

from pathlib import Path
from typing import Any

from hogfm.data.readers import stream_fasta_records


def fasta_to_hf_dataset(path: str | Path) -> Any:
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError("Install `datasets` to use HuggingFace dataset integration.") from exc
    rows = [{"name": record.name, "sequence": record.sequence} for record in stream_fasta_records(path)]
    return Dataset.from_list(rows)
