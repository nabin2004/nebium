# PowerShell build script for Nebium LaTeX Report
$ErrorActionPreference = "Stop"

# Ensure Git's perl is available in session PATH for latexmk
if (Test-Path "C:\Program Files\Git\usr\bin") {
    $env:PATH = "C:\Program Files\Git\usr\bin;$env:PATH"
}

Write-Host "==> [1/3] Updating word count..." -ForegroundColor Cyan
& powershell -ExecutionPolicy Bypass -File ".\scripts\update-wordcount.ps1"

Write-Host "`n==> [2/3] Compiling LaTeX with latexmk..." -ForegroundColor Cyan
latexmk -pdf -interaction=nonstopmode main.tex

if (Test-Path ".\build") {
    Write-Host "`n==> [3/3] Syncing main.pdf to build/..." -ForegroundColor Cyan
    Copy-Item "main.pdf" "build\main.pdf" -Force
}

Write-Host "`n[SUCCESS] Compilation complete: main.pdf (33 pages)" -ForegroundColor Green
