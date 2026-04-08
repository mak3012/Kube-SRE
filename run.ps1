param(
  [int]$Port = 8001
)

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

Write-Host "== Kube-SRE-Gym local run ==" -ForegroundColor Cyan
Write-Host "Port: $Port"

$env:HOST = "127.0.0.1"
$env:PORT = "$Port"

Write-Host ""
Write-Host "Starting server at http://127.0.0.1:$Port/docs" -ForegroundColor Green
python app.py

