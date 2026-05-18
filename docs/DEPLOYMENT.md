# Deployment

## Local API

```bash
docker compose up --build
```

The API is available at:

```text
http://localhost:8000
```

## Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
```

For GPU nodes:

```bash
kubectl label node <node-name> accelerator=nvidia
kubectl patch deployment hogfm-api --patch-file k8s/gpu-nodepool-patch.yaml
```

## Checkpoints

Set `HOGFM_CHECKPOINT` to a classifier checkpoint produced by:

```bash
python -m hogfm.training.train --train train.jsonl --valid valid.jsonl
```

Without a checkpoint, the service starts with a randomly initialized model for integration testing.
