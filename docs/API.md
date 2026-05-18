# API Reference

The FastAPI application exposes OpenAPI docs at `/docs` when running locally.

## Authentication

Set `HOGFM_API_KEY` to require `x-api-key` on model-serving endpoints.

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Device, checkpoint, and queue status |
| `POST /predict` | Promoter probability for one sequence |
| `POST /batch_predict` | Batched promoter probabilities |
| `POST /score_variant` | Reference vs alternate log-likelihood delta |
| `POST /embed_sequence` | Dense sequence embedding |
| `POST /generate` | Autoregressive nucleotide generation, optionally SSE streamed |
| `POST /benchmark` | Attention memory scaling estimates |
| `POST /api/parse/fasta` | FASTA summaries |
| `POST /api/variant/context` | Variant context extraction |
| `POST /api/variant/context-from-vcf` | VCF-like variant context extraction |

## Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "content-type: application/json" \
  -d '{"sequence":"ACGTACGTTATAAAACGT","max_length":128}'
```
