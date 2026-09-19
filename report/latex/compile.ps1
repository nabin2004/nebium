<#
.SYNOPSIS
    Compiles the Nebium LaTeX Research Report (CMP6232 / CMP6228).

.DESCRIPTION
    Automates the full LaTeX compilation pipeline for main.tex:
    1. Calculates main-body word count across sections and updates wordcount.tex.
    2. Runs pdflatex (Pass 1).
    3. Runs biber for BCU Harvard references (BibLaTeX).
    4. Runs pdflatex (Pass 2 & Pass 3) to finalize TOC, figures, tables, and citations.
    5. Optionally cleans auxiliary files or opens the generated PDF.

.PARAMETER Quick
    Fast single-pass compilation (skips Biber and auxiliary passes). Ideal for quick preview while drafting.

.PARAMETER NoBib
    Runs 2 pdflatex passes for cross-references but skips Biber (if citations haven't changed).

.PARAMETER Clean
    Removes auxiliary build files (.aux, .bbl, .bcf, .blg, .log, .out, .run.xml, .synctex.gz, .toc, .lof, .lot).

.PARAMETER Open
    Launches the compiled main.pdf in the default PDF viewer upon completion.

.PARAMETER SkipWordCount
    Skips recalculating and updating wordcount.tex.

.EXAMPLE
    .\compile.ps1
    Full 3-pass compilation with Biber and automated word count.

.EXAMPLE
    .\compile.ps1 -Quick -Open
    Fast single-pass compilation and immediately open the PDF.

.EXAMPLE
    .\compile.ps1 -Clean
    Cleans auxiliary cache files and recompiles from scratch.
#>

[CmdletBinding()]
param(
    [switch]$Quick,
    [switch]$NoBib,
    [switch]$Clean,
    [switch]$Open,
    [switch]$SkipWordCount
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Ensure working directory is the latex folder where this script resides
$LatexDir = $PSScriptRoot
Push-Location $LatexDir

function Write-Step {
    param([string]$Message)
    Write-Host "`n[Nebium Build] " -ForegroundColor Cyan -NoNewline
    Write-Host $Message -ForegroundColor Yellow
}

function Write-Success {
    param([string]$Message)
    Write-Host "[Nebium Build] " -ForegroundColor Cyan -NoNewline
    Write-Host $Message -ForegroundColor Green
}

function Write-Fail {
    param([string]$Message)
    Write-Host "[Nebium Build ERROR] " -ForegroundColor Red -NoNewline
    Write-Host $Message -ForegroundColor Red
}

# ---------------------------------------------------------------------------
# Clean auxiliary files if requested
# ---------------------------------------------------------------------------
$AuxExtensions = @(".aux", ".bbl", ".bcf", ".blg", ".log", ".out", ".run.xml", ".synctex.gz", ".toc", ".lof", ".lot")

if ($Clean) {
    Write-Step "Cleaning auxiliary files..."
    foreach ($ext in $AuxExtensions) {
        Get-ChildItem -Path $LatexDir -Filter "*$ext" -Recurse -File | ForEach-Object {
            Remove-Item $_.FullName -Force
            Write-Host "  Removed: $($_.Name)" -ForegroundColor DarkGray
        }
    }
    Write-Success "Clean complete."
}

# ---------------------------------------------------------------------------
# Calculate Main Body Word Count
# ---------------------------------------------------------------------------
if (-not $SkipWordCount) {
    Write-Step "Calculating main-body word count across core sections..."
    $CoreSections = @(
        "02_introduction.tex",
        "00_tools.tex",
        "03_problem_statement.tex",
        "04_proposed_method.tex",
        "05_experiments.tex",
        "06_summary.tex"
    )

    $TotalWords = 0
    $SectionBreakdown = @()

    foreach ($sec in $CoreSections) {
        $secPath = Join-Path $LatexDir "sections\$sec"
        $words = 0
        if (Test-Path $secPath) {
            $raw = Get-Content $secPath -Raw -Encoding UTF8
            # Remove LaTeX comments
            $stripped = $raw -replace '(?m)%.*$', ''
            # Remove common LaTeX macros and commands
            $stripped = $stripped -replace '\\[a-zA-Z]+(\[[^\]]*\])?(\{[^\}]*\})?', ' '
            # Count alphanumeric words
            $matches = [regex]::Matches($stripped, '\b[a-zA-Z0-9_\-]+\b')
            $words = $matches.Count
            $TotalWords += $words
        }
        $SectionBreakdown += [PSCustomObject]@{
            Section = $sec
            Words   = $words
        }
    }

    $SectionBreakdown | Format-Table -AutoSize | Out-String | Write-Host -ForegroundColor Gray
    Write-Host "Total Main Body Word Count: " -NoNewline -ForegroundColor White
    Write-Host "$TotalWords words " -NoNewline -ForegroundColor Green
    Write-Host "(Target Limit: 3000 + 10% = 3300 max)`n" -ForegroundColor DarkCyan

    # Update wordcount.tex
    $WordCountFile = Join-Path $LatexDir "wordcount.tex"
    Set-Content -Path $WordCountFile -Value "\newcommand{\ReportWordCount}{$TotalWords}" -Encoding UTF8
    Write-Success "Updated wordcount.tex with \ReportWordCount{$TotalWords}"
}

# ---------------------------------------------------------------------------
# Verify Required Executables
# ---------------------------------------------------------------------------
if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue)) {
    Write-Fail "pdflatex was not found in PATH. Please verify your MiKTeX installation."
    Pop-Location
    exit 1
}

# ---------------------------------------------------------------------------
# Compilation Passes
# ---------------------------------------------------------------------------
try {
    # PASS 1: Initial pdflatex run
    Write-Step "Pass 1: pdflatex (generating aux & bcf files)..."
    & pdflatex -interaction=nonstopmode -synctex=1 main.tex
    if ($LASTEXITCODE -ne 0) {
        throw "Pass 1 pdflatex failed with exit code $LASTEXITCODE. Check main.log for details."
    }

    if ($Quick) {
        Write-Success "Quick pass completed."
    }
    else {
        # BIBER: Bibliography resolution
        if (-not $NoBib) {
            if (Get-Command biber -ErrorAction SilentlyContinue) {
                Write-Step "Biber: Processing bibliography (BCU Harvard / biblatex)..."
                & biber main
                if ($LASTEXITCODE -ne 0) {
                    Write-Host "Biber returned exit code $LASTEXITCODE (Check main.blg if references fail)." -ForegroundColor DarkYellow
                } else {
                    Write-Success "Biber bibliography processed successfully."
                }
            } else {
                Write-Host "Warning: biber not found in PATH. Skipping bibliography compilation." -ForegroundColor Yellow
            }
        }

        # PASS 2: Integrate references and citations
        Write-Step "Pass 2: pdflatex (resolving cross-references and citations)..."
        & pdflatex -interaction=nonstopmode -synctex=1 main.tex
        if ($LASTEXITCODE -ne 0) {
            throw "Pass 2 pdflatex failed with exit code $LASTEXITCODE."
        }

        # PASS 3: Finalize TOC, LOF, LOT, and numbering
        Write-Step "Pass 3: pdflatex (finalizing table of contents and page numbers)..."
        & pdflatex -interaction=nonstopmode -synctex=1 main.tex
        if ($LASTEXITCODE -ne 0) {
            throw "Pass 3 pdflatex failed with exit code $LASTEXITCODE."
        }
    }

    $PdfPath = Join-Path $LatexDir "main.pdf"
    if (Test-Path $PdfPath) {
        $pdfSize = [math]::Round(((Get-Item $PdfPath).Length / 1MB), 2)
        Write-Host "`n========================================================" -ForegroundColor Green
        Write-Success "Compilation successful! Output: main.pdf ($pdfSize MB)"
        Write-Host "========================================================`n" -ForegroundColor Green

        if ($Open) {
            Write-Step "Opening main.pdf in default viewer..."
            Start-Process $PdfPath
        }
    } else {
        throw "main.pdf was not found after compilation."
    }
}
catch {
    Write-Fail $_.Exception.Message
    Pop-Location
    exit 1
}
finally {
    Pop-Location
}
