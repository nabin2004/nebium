<#
.SYNOPSIS
    Convenience wrapper to compile the Nebium LaTeX report from workspace root.
#>

[CmdletBinding()]
param(
    [switch]$Quick,
    [switch]$NoBib,
    [switch]$Clean,
    [switch]$Open,
    [switch]$SkipWordCount
)

$Script = Join-Path $PSScriptRoot "report\latex\compile.ps1"
if (-not (Test-Path $Script)) {
    Write-Error "Could not find compile.ps1 at $Script"
    exit 1
}

& $Script @PSBoundParameters
exit $LASTEXITCODE
