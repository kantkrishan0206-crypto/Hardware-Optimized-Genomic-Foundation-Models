$ErrorActionPreference = "Stop"

function Invoke-DockerChecked {
  param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$DockerArgs
  )
  & docker @DockerArgs
  if ($LASTEXITCODE -ne 0) {
    throw "docker $($DockerArgs -join ' ') failed with exit code $LASTEXITCODE"
  }
}

$containerStarted = $false

try {
  Invoke-DockerChecked build -t hogfm-api:smoke .
  Invoke-DockerChecked run --rm -d --name hogfm-smoke -p 8000:8000 hogfm-api:smoke
  $containerStarted = $true
  Start-Sleep -Seconds 10
  Invoke-RestMethod -Uri http://localhost:8000/health
} finally {
  if ($containerStarted) {
    & docker stop hogfm-smoke
  }
}
