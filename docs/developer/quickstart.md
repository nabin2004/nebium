# Nebium Developer Quickstart

Welcome to the Nebium codebase! This guide will help you set up your development environment and get familiar with the core workflows.

## Prerequisites
- **OS**: Linux, Windows, or macOS
- **Python**: 3.12+
- **Environment Manager**: [uv](https://github.com/astral-sh/uv) (Extremely fast Python package installer and resolver)

## 1. Setup

Clone the repository and install dependencies using `uv`:

```bash
git clone https://github.com/nabin2004/nebium.git
cd nebium
uv sync
```

This will create a virtual environment (`.venv`) and install all required packages defined in `pyproject.toml`.

## 2. Running Training Locally

The entry point for training is `scripts/train.py`. The project uses **Hydra** for configuration management.

To run a tiny stub model (useful for testing the pipeline):
```bash
uv run python scripts/train.py model=nebium_stub
```

To run a larger configuration (e.g., 117M parameters):
```bash
uv run python scripts/train.py model=nebium_117m
```

Configs are located in `configs/model/*.yaml`. 

## 3. The Core Directories

- **`src/models/transformer/`**: Contains the core architecture (`nebium.py` and `modules.py`).
- **`src/training/loop.py`**: The main PyTorch training loop, featuring gradient accumulation, AMP (mixed precision), and metric logging.
- **`src/data/`**: Data processing scripts (PGN -> UCI parsing, tokenization, dataset preparation).
- **`scripts/`**: Executable scripts for training, evaluation, and analysis.

## 4. Evaluation Suite

To test model strength against Stockfish or opening theory, run any of the evaluation scripts in the `scripts/` directory.

Example:
```bash
uv run python scripts/eval_self_play.py --depth 10 --games 5
```

For more details, see the [Evaluation Guide](evaluation.md).

## 5. Inference / API Deployment

Nebium ships with a FastAPI application to serve predictions.

```bash
uv run python src/api/app.py
```

Then visit `http://127.0.0.1:8000/docs` to test the API endpoints interactively. For more info, see the [Deployment Guide](deployment.md).
