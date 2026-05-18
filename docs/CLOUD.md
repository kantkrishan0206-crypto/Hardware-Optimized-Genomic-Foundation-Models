# Cloud Deployment

This repo includes provider templates rather than live credentials.

## AWS

Use `cloud/aws/ecs-task.json` for ECS on GPU-backed EC2 instances behind an ALB/API Gateway.

## GCP

Use `cloud/gcp/cloud-run-gpu.yaml` as a Cloud Run GPU service template.

## Azure

Use `cloud/azure/container-app.yaml` as an Azure Container Apps template.

## RunPod

Use `cloud/runpod/template.json` as a RunPod serverless or pod template.

## Production Notes

- Put checkpoints in a mounted object-store volume.
- Set `HOGFM_API_KEY` for authenticated endpoints.
- Use autoscaling only after benchmark-driven latency targets are established.
- Use provider GPU monitoring plus API queue-depth logs for scaling signals.
