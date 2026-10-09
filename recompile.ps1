<#
.SYNOPSIS
    Fast one-command recompilation script for the Nebium LaTeX Research Report.

.DESCRIPTION
    Runs word count updating and dual-pass pdflatex for cross-references.
    By default runs in fast mode (-NoBib). Pass -Full to run complete Biber bibliography.

.EXAMPLE
    .\recompile.ps1
    Fast 2-pass compilation with updated word count.

.EXAMPLE
    .\recompile.ps1 -Full
    Full 3-pass compilation including Biber bibliography.

.EXAMPLE
    .\recompile.ps1 -Open
    Recompiles and immediately opens the PDF in default viewer.
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

$Script = Join-Path $PSScriptRoot "report\latex\compile.ps1"
if (-not (Test-Path $Script)) {
    Write-Error "Could not find compile.ps1 at $Script"
    exit 1
}

& $Script @params
exit $LASTEXITCODE
