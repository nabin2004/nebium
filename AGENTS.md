# Antigravity Workspace Configuration (Nebium)

This document defines repository standards, architectural invariants, developer workflows, and writing guidelines for AI agents working in the **Nebium** repository.

---

## 1. Project Overview & Architecture

Nebium is a from-scratch causal Transformer engineered for self-supervised next-move prediction on chess games, developed as a Level 6 research project (CMP6232 / CMP6228). It processes space-separated UCI move sequences without board representation heuristics, evaluating emergent chess understanding across scale.

### Core Architecture Invariants
- **Model Topology (`src/models/transformer/`):**
  - Pure causal Transformer with Rotary Position Embeddings (RoPE) applied to queries and keys in self-attention.
  - Pre-normalization using RMSNorm (`norm.py`).
  - SwiGLU feed-forward networks (`ffn.py`).
  - Causal autoregressive masking paired with padding masking (loss ignores target index `-100`).
  - Full model assembly in `nebium.py`.
- **Model Scaling Family:**
  - `nebium_stub` (~8K params, 64-d, 1 layer) – Fast test & fixture execution.
  - `nebium_base` (~6M params, 512-d, 6 layers) – Local prototyping.
  - `nebium_117m` (117M params, GPT-2 Small scale).
  - `nebium_345m` (345M params, GPT-2 Medium scale).
  - `nebium_762m` (762M params, GPT-2 Large scale, published as `nabin2004/nebium-large`).
  - `nebium_1_5b` (1.5B params, GPT-2 XL scale).
- **Design Principles:**
  - **Configuration over Code:** All parameters managed via Hydra configs under `configs/`. Do not hardcode model dimensions, batch sizes, or paths.
  - **Decoupled Architecture:** The training loop (`src/training/trainer.py`) interacts with pluggable dataset readers and logging interfaces, never directly coupling to PGN files or W&B.
  - **Export Compatibility:** Checkpoints export to GGUF format (`src/export/gguf.py`) and Hugging Face Hub (`src/hub/push.py`). Preserve weight names and export schemas.

---

## 2. Directory Layout & Key Modules

| Directory / File | Description |
|---|---|
| `src/models/transformer/` | RoPE, RMSNorm, SwiGLU FFN, Causal Attention, Transformer Block, and Nebium model. |
| `src/data/` | Streaming PGN/zst reader, UCI parser, BPE tokenizer, and corpus preparation (`prepare.py`). |
| `src/training/` | Decoupled training engine, learning rate schedulers, metrics, and checkpointing. |
| `src/evaluation/` | Next-move accuracy, top-5 accuracy, perplexity, puzzle solver, and scaling law evaluation. |
| `src/export/` | GGUF quantization and model conversion for local inference. |
| `src/interpretability/` | Probing classifiers, board state decoders, and attention pattern extraction. |
| `src/hub/` | Hugging Face model/dataset push utilities. |
| `src/utils/kaggle.py` | Kaggle API synchronization, notebook generation, and artifact retrieval. |
| `nebium/` | Public top-level package API (`load_model`, `predict_next_moves`, `nebium.utils`). |
| `nebium_scope/` | Editable intelligence & interpretability toolkit (Gradio UI, activation steering, logit lens). |
| `dev_docs/` | Developer journey, engineering journal, and Architecture Decision Records (ADRs). |
| `mkdocs.yml` | MkDocs Material configuration building static docs into `docs/dev_journey/`. |
| `docs/NEBIUM_SCOPE.md` | Comprehensive developer documentation & API reference for NebiumScope. |
| `configs/` | Hydra configuration tree (`config.yaml`, `model/`, `training/`, `data/`, `logging/`, `hub/`). |
| `data/fixtures/` | Checked-in sample PGN for cheap unit and integration tests. |
| `report/latex/` | Full academic research paper (main.tex, sections, references, figures). |
| `compile_report.ps1` | PowerShell wrapper compiling the LaTeX research report with word count & Biber. |
| `scripts/` | Standalone execution scripts (training, data prep, scaling analysis, Hugging Face deploy). |
| `tests/` | Pytest test suite covering smoke tests, tokenization, GGUF export, and integration. |

---

## 3. Tooling & Development Workflow

The project uses Python 3.10+ (typically 3.12) managed via **`uv`**.

### Essential Commands
- **Remote / Kaggle Installation via Git:**
  ```bash
  pip install git+https://github.com/nabin2004/nebium.git
  ```
  Programmatic usage:
  ```python
  import nebium
  model, tokenizer = nebium.load_model("large")  # 'small', 'base', 'medium', 'large'
  continuation = model.generate_moves("e2e4 e7e5", tokenizer)

  # Move debugging and spatial board features:
  diag = nebium.utils.debug_position("e2e4 e7e5", model=model, tokenizer=tokenizer)
  tensor = nebium.utils.board_to_tensor("e2e4 e7e5")  # (12, 8, 8) piece planes
  ```
- **Environment Sync (Local Repo):**
  ```powershell
  uv sync
  ```
- **Run Fast Smoke Tests (Validates whole pipeline in ~5s):**
  ```powershell
  uv run pytest tests/test_smoke_test.py
  python scripts/train.py --config-name kaggle --smoke-test
  ```
- **Run Full Pytest Suite:**
  ```powershell
  uv run pytest
  # Skip integration tests if needed:
  uv run pytest -m "not integration"
  ```
- **Launch NebiumScope Gradio Dashboard:**
  ```powershell
  python -m nebium_scope.app --port 7860
  # Or programmatically: python -c "import nebium_scope as ns; ns.launch()"
  ```
- **Preview Developer Journey & MkDocs Docs (Live Reload):**
  ```powershell
  uv run mkdocs serve
  # Build static production site to docs/dev_journey/:
  uv run mkdocs build
  ```
- **Train Locally (Without external logging):**
  ```powershell
  python scripts/train.py logging=disabled
  ```
- **Train on Lichess Corpus:**
  ```powershell
  python scripts/train.py data=lichess model=nebium_base training=default logging=disabled
  ```
- **Evaluate Scaling Laws:**
  ```powershell
  uv run python scripts/eval_scaling_laws.py --small checkpoint_small.pt --medium checkpoint_medium.pt --large checkpoint_large.pt --data-path data/fixtures/sample.pgn --report paper_assets/SCALING_LAWS.md
  ```
- **Compile LaTeX Research Paper (CMP6232):**
  ```powershell
  .\compile_report.ps1
  # Fast preview pass:
  .\compile_report.ps1 -Quick -Open
  # Clean build artifacts:
  .\compile_report.ps1 -Clean
  ```

---

## 4. Coding Standards & Invariants

1. **Hydra Overrides:** When adding parameters, define them in the appropriate config under `configs/` with sensible defaults. Use CLI overrides (`key=value`) for runs.
2. **PyTorch Best Practices:**
   - Always vectorize tensor operations; avoid iterating over batches with Python loops.
   - Use `ignore_index=-100` for padded tokens in causal cross-entropy loss.
   - Ensure device agnosticism (`device = torch.device(...)` or using tensor `.to(device)`).
3. **Data & Artifact Hygiene:**
   - Do **NOT** commit raw dataset files (`.zst`, heavy `.pgn`), `.venv`, `wandb/`, `outputs/`, or checkpoint weights to git.
   - Use `data/fixtures/sample.pgn` for automated tests and CI.
4. **Backward Compatibility:**
   - Do not alter existing checkpoint layer naming in `Nebium` without maintaining migration/loading compatibility in `load_checkpoint`.
   - Keep GGUF export (`src/export/gguf.py`) aligned with the tensor layout expected by llama.cpp / GGUF readers.

---

## 5. Writing & Academic Integrity Guidelines (CMP6232)

When drafting, reviewing, or editing research papers, technical reports, and documentation:
- **Mandatory Skill Compliance:** Adhere to the `avoid-ai-writing` skill located at `.agents/skills/avoid-ai-writing/SKILL.md` (and globally at `~/.gemini/config/skills/avoid-ai-writing/SKILL.md`).
- **Forbidden AI Vocabulary & Phrasing:** Consult `references/patterns.md` for banned AI-isms and inflated buzzwords. Specifically avoid:
  - *"delve", "crucial", "testament", "tapestry", "beacon", "groundbreaking", "transformative", "pivotal", "underscores", "interplay", "holistic", "seamlessly", "fosters"*
  - Superficial rule-of-three phrasing (e.g., *"speed, accuracy, and efficiency"*).
  - Empty transitional scaffolding (*"Furthermore, it is worth noting that...", "In the ever-evolving landscape of..."*).
- **Evidence-Backed Prose:** All claims must be concrete, quantitative, and directly supported by experimental data (e.g., exact parameter counts, cross-entropy loss values, perplexity, puzzle pass rates, Elo scores, scaling coefficients).
- **Academic Standards:** Ensure all technical prose satisfies Level 6 assessment criteria (CMP6232 Machine Learning & Neural Computing) with proper BCU Harvard citations (`\parencite`, `\textcite`) managed via BibLaTeX/Biber.

---

## 6. Agent Verification Protocol

Before completing any task:
1. **Code Changes:** Run `uv run pytest tests/test_smoke_test.py` (or targeted test files) to verify that modifications do not break the pipeline.
2. **Report Changes:** If modifying files in `report/latex/`, run `.\compile_report.ps1 -Quick` or `.\compile_report.ps1` to ensure LaTeX builds cleanly without unresolved references or syntax errors.
3. **Precision:** Make minimal, surgical edits. Preserve unrelated docstrings, comments, and structure.
