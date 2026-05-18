from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from hogfm.kernels.linear_attention import estimate_attention_memory
from hogfm.parsing.fasta import build_variant_context, parse_fasta, parse_vcf_line

app = FastAPI(
    title="Hardware-Optimized Genomic Foundation Models API",
    version="0.1.0",
    description="Sequence parsing and memory estimation APIs for long-context genomic models.",
)


class FastaParseRequest(BaseModel):
    text: str = Field(min_length=1)


class FastaRecordResponse(BaseModel):
    name: str
    length: int
    gc_fraction: float


class VariantContextRequest(BaseModel):
    fasta_text: str = Field(min_length=1)
    contig: str | None = None
    position: int = Field(gt=0)
    reference: str = Field(min_length=1)
    alternate: str = Field(min_length=1)
    flank: int = Field(default=512, ge=0, le=100000)


class VcfContextRequest(BaseModel):
    fasta_text: str = Field(min_length=1)
    vcf_line: str = Field(min_length=1)
    flank: int = Field(default=512, ge=0, le=100000)


class VariantContextResponse(BaseModel):
    contig: str
    position: int
    reference: str
    alternate: str
    start: int
    end: int
    reference_context: str
    alternate_context: str
    token_ids: list[int]


class MemoryEstimateRequest(BaseModel):
    tokens: int = Field(gt=0)
    heads: int = Field(gt=0)
    dim: int = Field(gt=0)
    dtype_bytes: int = Field(default=2, gt=0)


@app.middleware("http")
async def timing_middleware(request: Request, call_next: Any) -> Any:
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["x-hogfm-latency-ms"] = f"{elapsed_ms:.3f}"
    return response


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/parse/fasta", response_model=list[FastaRecordResponse])
async def parse_fasta_endpoint(payload: FastaParseRequest) -> list[FastaRecordResponse]:
    try:
        records = parse_fasta(payload.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    responses: list[FastaRecordResponse] = []
    for record in records:
        gc = record.sequence.count("G") + record.sequence.count("C")
        responses.append(
            FastaRecordResponse(
                name=record.name,
                length=record.length,
                gc_fraction=gc / record.length if record.length else 0.0,
            )
        )
    return responses


@app.post("/api/variant/context", response_model=VariantContextResponse)
async def variant_context(payload: VariantContextRequest) -> VariantContextResponse:
    try:
        records = parse_fasta(payload.fasta_text)
        if payload.contig is None:
            record = records[0]
        else:
            matches = [item for item in records if item.name == payload.contig]
            if not matches:
                raise ValueError(f"Contig not found in FASTA: {payload.contig}")
            record = matches[0]
        context = build_variant_context(
            record=record,
            position=payload.position,
            reference=payload.reference,
            alternate=payload.alternate,
            flank=payload.flank,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return VariantContextResponse(**context.__dict__)


@app.post("/api/variant/context-from-vcf", response_model=VariantContextResponse)
async def variant_context_from_vcf(payload: VcfContextRequest) -> VariantContextResponse:
    try:
        contig, position, reference, alternate = parse_vcf_line(payload.vcf_line)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    request = VariantContextRequest(
        fasta_text=payload.fasta_text,
        contig=contig,
        position=position,
        reference=reference,
        alternate=alternate,
        flank=payload.flank,
    )
    return await variant_context(request)


@app.post("/api/estimate-memory")
async def memory_estimate(payload: MemoryEstimateRequest) -> dict[str, float | int]:
    estimate = estimate_attention_memory(
        tokens=payload.tokens,
        heads=payload.heads,
        dim=payload.dim,
        dtype_bytes=payload.dtype_bytes,
    )
    return {
        "tokens": estimate.tokens,
        "heads": estimate.heads,
        "dtype_bytes": estimate.dtype_bytes,
        "quadratic_attention_bytes": estimate.quadratic_attention_bytes,
        "linear_attention_bytes": estimate.linear_attention_bytes,
        "reduction_factor": estimate.reduction_factor,
    }
