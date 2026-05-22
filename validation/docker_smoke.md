# Docker Smoke Validation

## 2026-05-22 Local Result

The Docker CLI is installed, but Docker Desktop's Linux engine was not reachable from this
workspace during validation.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\docker_smoke.ps1
```

Result:

```text
docker build -t hogfm-api:smoke . failed with exit code 1
failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine
```

The smoke script now exits non-zero on Docker build or runtime failure. Re-run the command after
starting Docker Desktop or on a Linux CI runner with Docker available.
