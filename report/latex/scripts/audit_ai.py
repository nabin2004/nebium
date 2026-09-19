import re

with open('report/latex/sections/05_experiments.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# Let's count words in each subsection's body prose directly
sections = [
    ("5.1 Training Configurations", text[text.find(r"\subsection{Training Configurations"):text.find(r"\subsection{Empirical Scaling")]),
    ("5.2 Empirical Scaling", text[text.find(r"\subsection{Empirical Scaling"):text.find(r"\subsection{Move Accuracy")]),
    ("5.3 Move Accuracy & Legality", text[text.find(r"\subsection{Move Accuracy"):text.find(r"\subsection{Validation Benchmarks")]),
    ("5.4 Validation Benchmarks", text[text.find(r"\subsection{Validation Benchmarks"):text.find(r"\subsection{Comparative Baseline")]),
    ("5.5 Comparative Baseline", text[text.find(r"\subsection{Comparative Baseline"):text.find(r"\subsection{Qualitative Move")]),
    ("5.6 Qualitative Move Analysis", text[text.find(r"\subsection{Qualitative Move"):])
]

print("=== ACCURATE PROSE WORD COUNTS ===")
total_words = 0
for name, sec_text in sections:
    # strip docstrings / comments
    lines = [l for l in sec_text.splitlines() if not l.strip().startswith('%')]
    body = '\n'.join(lines)
    # strip figures, tables, math blocks, centers
    body = re.sub(r'\\begin\{figure\*?\}[\s\S]*?\\end\{figure\*?\}', '', body)
    body = re.sub(r'\\begin\{table\*?\}[\s\S]*?\\end\{table\*?\}', '', body)
    body = re.sub(r'\\begin\{center\}[\s\S]*?\\end\{center\}', '', body)
    body = re.sub(r'\\subsection\{.*?\}', '', body)
    body = re.sub(r'\\label\{.*?\}', '', body)
    # strip simple inline commands keeping the text inside
    body = re.sub(r'\\(?:textbf|textit|texttt|textcite|parencite|ref)\{([^}]*)\}', r'\1', body)
    # strip remaining latex commands
    body = re.sub(r'\\[a-zA-Z]+', ' ', body)
    words = re.findall(r'\b[A-Za-z0-9_\-\./\$\\]+\b', body)
    # filter purely latex artifacts
    clean_words = [w for w in words if not w.startswith('\\') and not w in ['b', 'linewidth', 'H', 'c']]
    wc = len(clean_words)
    total_words += wc
    print(f"{name}: {wc} words")

print(f"\nTotal Section 5 Body Words: {total_words}")
