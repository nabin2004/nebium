# Deployment and Inference

This document outlines how Nebium models are packaged and deployed for inference.

## 1. GGUF Export (Llama.cpp Integration)
At the end of training, Nebium attempts to export the PyTorch `.pt` weights into the GGUF format using `scripts/export_gguf.py`.
- **Why GGUF?** The GGUF format allows for highly optimized inference (including quantization) via engines like `llama.cpp`. This makes it viable to run Nebium smoothly on CPUs or edge devices.
- The exported `.gguf` file encapsulates the model weights, hyperparameter configuration, and tokenizer vocabulary.

## 2. Hugging Face Hub Integration
After training (and optionally GGUF export), `scripts/push_to_hf.py` is invoked to push the model artifacts to a Hugging Face repository.
- Controlled via `cfg.hf_hub.push = true`.
- Pushes the raw `best_model.pt`, the PyTorch checkpoint `checkpoint.pt`, and any `.gguf` variants.
- Uploads the tokenizer configuration (`tokenizer.json`).

## 3. The FastAPI Server
**Files:** `src/api/app.py`

Nebium includes a production-ready FastAPI application for serving the model via HTTP endpoints.

### Setup
```bash
uv run python src/api/app.py
```
This spawns a Uvicorn server on `0.0.0.1:8000`.

### Endpoints
- **`GET /health`**: Health check.
- **`POST /generate`**: The primary inference endpoint. 
  - **Payload:** `{"fen": "rnbqkbnr/...", "moves": "e2e4 e7e5", "max_moves": 1, "temperature": 0.0}`
  - The API initializes a `python-chess` board with the FEN, pushes the sequence of moves, and then prompts the Nebium model to generate the next `max_moves`.
  - The endpoint natively utilizes the `legal_tokens_filter` function defined in `benchmark_latency.py` to ensure the API never returns an invalid chess move.

### 4. Interactive Gradio UI
Nebium provides a full-featured visual chessboard and generation interface in [`scripts/app.py`](../../scripts/app.py) (also accessible at root [`app.py`](../../app.py)).

- **Model Family Selection:** Seamlessly switch between Nebium-Small (117M), Nebium-Medium (345M), Nebium-Large (762M), Nebium-Base (6M), and local checkpoints (`best_model.pt`).
- **Interactive SVG Board:** Real-time visual board rendering with last-move highlights, SAN move history, FEN export, and game status.
- **Rule Enforcement:** Optional real-time legal move filtering guaranteeing valid moves.

#### Local Execution
```bash
# Run locally on http://127.0.0.1:7860
uv run python scripts/app.py

# Launch with a public shareable HTTPS link:
uv run python scripts/app.py --share
```

#### Hugging Face Spaces Deployment via CLI
To deploy the interactive web UI directly to Hugging Face Spaces:
```bash
# Package and deploy to Hugging Face Spaces:
uv run python scripts/deploy_space.py --repo-id nabin2004/nebium-chess
```

