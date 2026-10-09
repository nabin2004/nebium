<#
.SYNOPSIS
    Fast one-command recompilation script for the Nebium LaTeX Research Report.
    Works directly from inside the report/latex directory.
#>

[CmdletBinding()]
param(
    [switch]$Full,
    [switch]$Open,
    [switch]$Clean
)

$params = @{}
if (-not $Full) {
    $params["NoBib"] = $true
}
if ($Open) {
    $params["Open"] = $true
}
if ($Clean) {
    $params["Clean"] = $true
}

$Script = Join-Path $PSScriptRoot "compile.ps1"
if (-not (Test-Path $Script)) {
    Write-Error "Could not find compile.ps1 at $Script"
    exit 1
}

& $Script @params
exit $LASTEXITCODE
