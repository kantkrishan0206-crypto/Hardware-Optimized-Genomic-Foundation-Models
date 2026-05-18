# Tutorial

## 1. Prepare Data

```bash
python scripts/prepare_dataset.py --examples 256 --length 128
```

## 2. Train a Small Classifier

```bash
python -m hogfm.training.train \
  --train data/processed/promoter/train.jsonl \
  --valid data/processed/promoter/valid.jsonl \
  --epochs 1 \
  --hidden-size 64 \
  --layers 2
```

## 3. Serve

```bash
HOGFM_CHECKPOINT=checkpoints/promoter/model.safetensors \
uvicorn hogfm.api.main:app --host 0.0.0.0 --port 8000
```

## 4. Query

```bash
python examples/inference_client.py --sequence ACGTACGTTATAAAACGT
```
