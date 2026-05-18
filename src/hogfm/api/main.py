from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from hogfm.kernels.linear_attention import estimate_attention_memory
from hogfm.models import (
    GenomicFoundationModel,
    GenomicSequenceClassifier,
    GenomicTransformerConfig,
    sequence_log_likelihood,
)
from hogfm.observability import configure_logging
from hogfm.parsing.fasta import build_variant_context, parse_fasta, parse_vcf_line
from hogfm.tokenization import GenomicTokenizer

configure_logging()
LOGGER = logging.getLogger("hogfm.api")


class SequenceRequest(BaseModel):
    sequence: str = Field(min_length=1)
    max_length: int = Field(default=512, gt=8, le=65536)

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, value: str) -> str:
        return GenomicTokenizer.normalize(value)


class PredictResponse(BaseModel):
    predicted_label: int
    promoter_probability: float
    model_status: str


class EmbedSequenceResponse(BaseModel):
    embedding_dim: int
    embedding: list[float]
    model_status: str


class ScoreVariantRequest(BaseModel):
    reference_sequence: str = Field(min_length=1)
    alternate_sequence: str = Field(min_length=1)
    max_length: int = Field(default=512, gt=8, le=65536)

    @field_validator("reference_sequence", "alternate_sequence")
    @classmethod
    def validate_sequence(cls, value: str) -> str:
        return GenomicTokenizer.normalize(value)


class ScoreVariantResponse(BaseModel):
    reference_log_likelihood: float
    alternate_log_likelihood: float
    delta_log_likelihood: float
    interpretation: str
    model_status: str


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


@dataclass
class InferenceJob:
    kind: str
    payload: dict[str, Any]
    future: asyncio.Future[Any]


class InferenceService:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[InferenceJob] = asyncio.Queue(maxsize=128)
        self.worker_task: asyncio.Task[None] | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GenomicTokenizer(strategy="kmer", k=3)
        self.model_status = "randomly_initialized"
        self.classifier = self._build_classifier()
        self._load_checkpoint_if_configured()

    def _build_classifier(self) -> GenomicSequenceClassifier:
        config = GenomicTransformerConfig(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=96,
            num_layers=2,
            num_heads=4,
            intermediate_size=384,
            attention_backend="performer",
            performer_features=48,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        encoder = GenomicFoundationModel(config)
        return GenomicSequenceClassifier(encoder).to(self.device).eval()

    def _load_checkpoint_if_configured(self) -> None:
        checkpoint = os.getenv("HOGFM_CHECKPOINT")
        if not checkpoint:
            return
        path = Path(checkpoint)
        if not path.exists():
            LOGGER.warning("checkpoint_missing", extra={"hogfm_path": str(path)})
            return
        payload = torch.load(path, map_location=self.device)
        state_dict = payload.get("model_state_dict", payload)
        self.classifier.load_state_dict(state_dict, strict=False)
        self.model_status = f"loaded:{path.name}"
        LOGGER.info("checkpoint_loaded", extra={"hogfm_path": str(path)})

    async def start(self) -> None:
        self.worker_task = asyncio.create_task(self._worker(), name="hogfm-inference-worker")

    async def stop(self) -> None:
        if self.worker_task is not None:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

    async def submit(self, kind: str, payload: dict[str, Any]) -> Any:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        await self.queue.put(InferenceJob(kind=kind, payload=payload, future=future))
        return await future

    async def _worker(self) -> None:
        while True:
            job = await self.queue.get()
            try:
                result = await asyncio.to_thread(self._dispatch, job.kind, job.payload)
                job.future.set_result(result)
            except Exception as exc:
                job.future.set_exception(exc)
            finally:
                self.queue.task_done()

    def _dispatch(self, kind: str, payload: dict[str, Any]) -> Any:
        if kind == "predict":
            return self._predict(payload["sequence"], payload["max_length"])
        if kind == "embed":
            return self._embed(payload["sequence"], payload["max_length"])
        if kind == "score_variant":
            return self._score_variant(
                payload["reference_sequence"],
                payload["alternate_sequence"],
                payload["max_length"],
            )
        raise ValueError(f"Unknown inference job kind: {kind}")

    def _encode(self, sequence: str, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer.batch_encode([sequence], max_length=max_length)
        input_ids = torch.tensor(encoded.input_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(encoded.attention_mask, dtype=torch.long, device=self.device)
        return input_ids, attention_mask

    @torch.inference_mode()
    def _predict(self, sequence: str, max_length: int) -> PredictResponse:
        input_ids, attention_mask = self._encode(sequence, max_length)
        output = self.classifier(input_ids=input_ids, attention_mask=attention_mask)
        probability = torch.softmax(output.logits, dim=-1)[0, 1].item()
        return PredictResponse(
            predicted_label=int(probability >= 0.5),
            promoter_probability=probability,
            model_status=self.model_status,
        )

    @torch.inference_mode()
    def _embed(self, sequence: str, max_length: int) -> EmbedSequenceResponse:
        input_ids, attention_mask = self._encode(sequence, max_length)
        output = self.classifier(input_ids=input_ids, attention_mask=attention_mask)
        assert output.hidden_states is not None
        embedding = output.hidden_states[0].detach().cpu().tolist()
        return EmbedSequenceResponse(
            embedding_dim=len(embedding),
            embedding=embedding,
            model_status=self.model_status,
        )

    @torch.inference_mode()
    def _score_variant(
        self,
        reference_sequence: str,
        alternate_sequence: str,
        max_length: int,
    ) -> ScoreVariantResponse:
        ref_ids, ref_mask = self._encode(reference_sequence, max_length)
        alt_ids, alt_mask = self._encode(alternate_sequence, max_length)
        encoder = self.classifier.encoder
        ref_ll = sequence_log_likelihood(encoder, ref_ids, ref_mask)[0].item()
        alt_ll = sequence_log_likelihood(encoder, alt_ids, alt_mask)[0].item()
        delta = alt_ll - ref_ll
        interpretation = "alternate_more_likely" if delta >= 0 else "reference_more_likely"
        return ScoreVariantResponse(
            reference_log_likelihood=ref_ll,
            alternate_log_likelihood=alt_ll,
            delta_log_likelihood=delta,
            interpretation=interpretation,
            model_status=self.model_status,
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    service = InferenceService()
    app.state.inference_service = service
    await service.start()
    yield
    await service.stop()


app = FastAPI(
    title="Hardware-Optimized Genomic Foundation Models API",
    version="0.2.0",
    description="Long-context genomic modeling, parsing, and inference service.",
    lifespan=lifespan,
)


@app.middleware("http")
async def timing_middleware(request: Request, call_next: Any) -> Any:
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["x-hogfm-latency-ms"] = f"{elapsed_ms:.3f}"
    LOGGER.info(
        "request_complete",
        extra={
            "hogfm_path": request.url.path,
            "hogfm_method": request.method,
            "hogfm_latency_ms": round(elapsed_ms, 3),
            "hogfm_status_code": response.status_code,
        },
    )
    return response


def _service(request: Request) -> InferenceService:
    return request.app.state.inference_service


@app.get("/health")
async def health(request: Request) -> dict[str, int | str]:
    service = _service(request)
    return {
        "status": "ok",
        "device": str(service.device),
        "model_status": service.model_status,
        "queue_depth": service.queue.qsize(),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: SequenceRequest, request: Request) -> PredictResponse:
    return await _service(request).submit("predict", payload.model_dump())


@app.post("/embed_sequence", response_model=EmbedSequenceResponse)
async def embed_sequence(payload: SequenceRequest, request: Request) -> EmbedSequenceResponse:
    return await _service(request).submit("embed", payload.model_dump())


@app.post("/score_variant", response_model=ScoreVariantResponse)
async def score_variant(payload: ScoreVariantRequest, request: Request) -> ScoreVariantResponse:
    return await _service(request).submit("score_variant", payload.model_dump())


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
