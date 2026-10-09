# Reproducing Nebium Research Report Figures & WandB Artifacts

This document provides complete instructions for reproducing, synchronizing, and rendering all publication-grade figures in the **Nebium Research Report** (*CMP6232 / CMP6228 Assessment 2*).

All empirical diagrams are strictly grounded in live **Weights & Biases (WandB)** run telemetry from the `nabinoli2004-wiseyak/nebium` project.

---

## 1. Authentication & Security Policy

To prevent ungrounded, synthetic, or corrupted figures from entering the academic submission, the figure generation engine enforces strict authentication:

- **Token Requirement**: A valid Weights & Biases API key is mandatory.
- **Interactive Prompting**: If no token is detected, the script prompts the user interactively with masked input.
- **Fail-Fast Halting**: If the user provides an empty token, aborts, or provides invalid credentials, **execution is immediately halted with exit code `1`**. No placeholder or fabricated data is ever substituted.

### Obtaining a WandB Token
If you do not have your token handy, obtain it from your account at:
[https://wandb.ai/authorize](https://wandb.ai/authorize)

---

## 2. Quickstart Execution

### Option A: Interactive Execution (Recommended)
Run the generator from the repository root. If `WANDB_API_KEY` is not in your environment, you will be prompted securely:

```bash
python scripts/generate_paper_diagrams.py
```

### Option B: CLI Flag (Non-Interactive)
Provide your key directly on the command line:

```bash
python scripts/generate_paper_diagrams.py --api-key <YOUR_WANDB_API_KEY>
```

### Option C: Environment Variable
Export the token in your shell before running:

**Linux / macOS:**
```bash
export WANDB_API_KEY="wandb_v1_..."
python scripts/generate_paper_diagrams.py
```

**Windows PowerShell:**
```powershell
$env:WANDB_API_KEY = "wandb_v1_..."
python scripts/generate_paper_diagrams.py
```

---

## 3. CLI Options and Configuration

The generator script supports the following command-line flags:

| Flag | Default | Description |
|---|---|---|
| `--api-key` | `None` (prompts/env) | WandB API authorization token |
| `--entity` | `nabinoli2004-wiseyak` | WandB team or user entity |
| `--project` | `nebium` | WandB project repository name |
| `--output-dir` | `report/latex/figures` | Destination directory for generated vector/raster figures |

---

## 4. Report Figures Catalog & Provenance

The generator creates and synchronizes the following 7 empirical figures referenced in the LaTeX report:

| Report Figure | File Names | Paper Section | Source Run / Telemetry | Description |
|---|---|---|---|---|
| **Figure 7** | `hardware_and_optimization_hq.pdf`<br>`hardware_and_optimization_hq.png` | Section 4.1 | Dual T4 Hardware Logs (`lsw3ck24`, `xvja7e0t`, `b87goptx`) | 3-panel figure showing throughput (tokens/sec), peak VRAM per GPU against the 16 GB ceiling, and the 2,000-step warmup cosine schedule. |
| **Figure 8(a)** | `wandb_scaling_laws_hq.png` | Section 4.2 | WandB Run `b87goptx`<br>(artifact: `publication/scaling_laws`) | Chinchilla compute-optimal frontier ($D^* = 20N$) vs. the 12.0M fixed token budget across Small, Medium, and Large. |
| **Figure 8(b)** | `wandb_training_dynamics_hq.png` | Section 4.2 | WandB Run `b87goptx`<br>(artifact: `publication/training_dynamics`) | Nebium-Large (762M) cross-entropy loss drop ($7.407 \rightarrow 6.307$) and perplexity descent ($950 \rightarrow 435$) over 3 epochs. |
| **Figure 9(a)** | `wandb_accuracy_legality_hq.png` | Section 4.3 | WandB Run `b87goptx`<br>(artifact: `publication/accuracy_and_legality`) | Nebium-Large next-token prediction accuracy (Top-1, Top-5) and rapid board-legal move rate ascent to 80%. |
| **Figure 9(b)** | `puzzle_and_generation_benchmarks.pdf`<br>`puzzle_and_generation_benchmarks.png` | Section 4.3 & 4.4 | WandB Runs `lsw3ck24`, `b87goptx` (`eval/puzzle_benchmarks`) | Stratified tactical puzzle solve rates across rating tiers and autoregressive horizon legality progression. |
| **Figure 10** | `validation_benchmarks_hq.pdf`<br>`validation_benchmarks_hq.png` | Section 4.4 | WandB Runs `lsw3ck24`, `xvja7e0t`, `b87goptx` | 6-panel validation dashboard tracking Top-5 accuracy, perplexity, overall solve rates, and $<1500$, $1500$--$2000$, and $2000+$ Elo puzzles. |
| **Figure 11** | `italian_game_qualitative.pdf`<br>`italian_game_qualitative.png` | Section 4.6 | `python-chess` Rollout Engine | 3-panel tournament vector chessboard tracing Italian Game Giuoco Pianissimo development and king coordinate drift failure mode. |

---

## 5. Recompiling the LaTeX Research Report

Once figures have been updated or regenerated:

```powershell
cd report/latex
powershell -ExecutionPolicy Bypass -File compile.ps1
```

The compilation script automatically:
1. Calculates main-body word count across the core 5 sections and updates `wordcount.tex`.
2. Runs `pdflatex` $\rightarrow$ `biber` $\rightarrow$ `pdflatex` (2 passes) to resolve citations and cross-references.
3. Produces the final submission file: `report/latex/main.pdf`.
