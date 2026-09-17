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

### Gradio UI UI
There is an experimental Gradio interface provided in `scripts/app.py` for testing Transformer generations interactively.
