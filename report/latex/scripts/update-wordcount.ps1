# Word count script for Assessment 2 report
# Calculates body words across main sections: 02_introduction, 03_problem_statement, 04_proposed_method, 05_experiments, 06_summary
$sectionsDir = Join-Path $PSScriptRoot "..\sections"
$targetFiles = @(
    "00_tools.tex",
    "02_introduction.tex",
    "03_problem_statement.tex",
    "04_proposed_method.tex",
    "05_experiments.tex",
    "06_summary.tex"
)

$totalWords = 0
foreach ($file in $targetFiles) {
    $filePath = Join-Path $sectionsDir $file
    if (Test-Path $filePath) {
        $text = Get-Content $filePath -Raw
        # Remove comments
        $text = $text -replace '(?m)%.*$', ''
        # Remove table and figure environments (including captions and tabular content)
        $text = $text -replace '(?s)\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}', ' '
        # Remove display math equations and inline math
        $text = $text -replace '(?s)\\begin\{(equation|align)\*?\}.*?\\end\{\1\*?\}', ' '
        $text = $text -replace '\$[^$]*\$', ' '
        # Strip citation, label, and ref commands
        $text = $text -replace '\\(parencite|cite|ref|label)\*?(?:\[[^\]]*\])?\{[^}]*\}', ' '
        # Preserve inner text in formatting commands like \textbf{...}, \textit{...}, \texttt{...}, \emph{...}
        for ($i = 0; $i -lt 3; $i++) {
            $text = $text -replace '\\(textbf|textit|texttt|emph)\{([^{}]*)\}', '$2'
        }
        # Strip section commands keeping title
        $text = $text -replace '\\(section|subsection|subsubsection)\*?\{([^{}]*)\}', '$2'
        # Strip any remaining backslash commands
        $text = $text -replace '\\[a-zA-Z]+', ' '
        # Remove remaining braces, brackets, and LaTeX punctuation
        $text = $text -replace '[\{\}\[\]\(\)\\_~]', ' '
        # Split into words
        $words = $text -split '\s+' | Where-Object { $_ -match '[a-zA-Z0-9]' }
        $count = $words.Count
        $totalWords += $count
        Write-Host "$file : $count words"
    }
}

Write-Host "Total Body Word Count: $totalWords"
$outFile = Join-Path $PSScriptRoot "..\wordcount.tex"
"\newcommand{\ReportWordCount}{$totalWords}" | Set-Content $outFile -Encoding ASCII
Write-Host "Updated $outFile"
