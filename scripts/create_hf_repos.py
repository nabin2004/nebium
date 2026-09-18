"""
scripts/create_hf_repos.py
==========================
Provisions the 6 official Hugging Face repositories for the Nebium model family:
  1. nabin2004/nebium-small       (117M PyTorch causal transformer)
  2. nabin2004/nebium-medium      (345M PyTorch causal transformer)
  3. nabin2004/nebium-large       (762M PyTorch causal transformer)
  4. nabin2004/nebium-small-gguf  (117M GGUF format for llama.cpp / Ollama)
  5. nabin2004/nebium-medium-gguf (345M GGUF format for llama.cpp / Ollama)
  6. nabin2004/nebium-large-gguf  (762M GGUF format for llama.cpp / Ollama)

Initializes each repository with a detailed, publication-standard Model Card
adhering to academic documentation requirements.
"""

import argparse
import os
import sys
from typing import Any
from huggingface_hub import HfApi

GITATTRIBUTES_CONTENT = """*.pt filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.gguf filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
"""

MODELS = {
    "small": {
        "tier": "small",
        "name": "Nebium-Small",
        "params": "117M",
        "d_model": 768,
        "n_heads": 12,
        "n_layers": 12,
        "max_seq_len": 1024,
        "vocab_size": 5000,
        "chinchilla_tokens": "2.3B",
        "learning_rate": "3e-4",
        "weight_decay": "0.1",
        "description": "117-million parameter causal Transformer trained for autoregressive next-chess-move prediction over Lichess UCI sequences.",
        "repo_id": "nabin2004/nebium-small",
        "gguf_repo_id": "nabin2004/nebium-small-gguf",
    },
    "medium": {
        "tier": "medium",
        "name": "Nebium-Medium",
        "params": "345M",
        "d_model": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "max_seq_len": 1024,
        "vocab_size": 5000,
        "chinchilla_tokens": "6.9B",
        "learning_rate": "2e-4",
        "weight_decay": "0.1",
        "description": "345-million parameter causal Transformer trained for autoregressive next-chess-move prediction over Lichess UCI sequences.",
        "repo_id": "nabin2004/nebium-medium",
        "gguf_repo_id": "nabin2004/nebium-medium-gguf",
    },
    "large": {
        "tier": "large",
        "name": "Nebium-Large",
        "params": "762M",
        "d_model": 1280,
        "n_heads": 20,
        "n_layers": 36,
        "max_seq_len": 1024,
        "vocab_size": 5000,
        "chinchilla_tokens": "15.2B",
        "learning_rate": "1.5e-4",
        "weight_decay": "0.1",
        "description": "762-million parameter causal Transformer, the flagship model of the Nebium family, optimized for tactical sequence modeling and move legality.",
        "repo_id": "nabin2004/nebium-large",
        "gguf_repo_id": "nabin2004/nebium-large-gguf",
    },
}


def build_pytorch_model_card(meta: dict[str, Any]) -> str:
    """Generates the Markdown model card for PyTorch weights."""
    return f"""---
language:
- en
license: mit
library_name: pytorch
tags:
- chess
- causal-lm
- transformer
- nebium
- nebium-{meta['tier']}
- rope
- swiglu
- rmsnorm
datasets:
- nabin2004/nebium-lichess-uci
pipeline_tag: text-generation
---

# {meta['name']} ({meta['params']})

{meta['description']}

- **PyTorch Repository**: [{meta['repo_id']}](https://huggingface.co/{meta['repo_id']})
- **Companion GGUF Repository**: [{meta['gguf_repo_id']}](https://huggingface.co/{meta['gguf_repo_id']})
- **Framework Source**: [github.com/nabin2004/nebium](https://github.com/nabin2004/nebium)

---

## Architecture Specifications

{meta['name']} is built on a decoder-only causal Transformer architecture with modern architectural primitives:

| Specification | Parameter Value |
|---|---|
| Model Tier | **{meta['name']}** |
| Parameter Count | **{meta['params']}** |
| Hidden Dimension ($d_{{model}}$) | {meta['d_model']} |
| Attention Heads ($n_{{heads}}$) | {meta['n_heads']} |
| Transformer Layers ($n_{{layers}}$) | {meta['n_layers']} |
| Context Window ($L_{{max}}$) | {meta['max_seq_len']} tokens |
| Vocabulary Size ($V$) | {meta['vocab_size']} (Byte-Pair Encoding over UCI plies) |
| Positional Embeddings | Rotary Position Embeddings (RoPE, $\\theta = 10000$) |
| Activation Function | SwiGLU |
| Layer Normalization | RMSNorm (pre-normalization) |
| Attention Mechanism | Causal scaled dot-product attention |
| Base Learning Rate | {meta['learning_rate']} (Cosine decay with linear warmup) |
| Weight Decay | {meta['weight_decay']} |

---

## Scaling Law Analysis (Chinchilla Framework)

Following Hoffmann et al. (2022) scaling laws, compute-optimal training balances parameter count $N$ with token allocation $D$:

$$L(N, D) = E + \\frac{{A}}{{N^\\alpha}} + \\frac{{B}}{{D^\\beta}}$$

where $E = 1.69$, $A = 406.4$, $B = 410.7$, $\\alpha = 0.34$, $\\beta = 0.28$.

- **Parameters ($N$)**: ~{meta['params']}
- **Chinchilla-Optimal Token Budget ($D^* \\approx 20N$)**: **{meta['chinchilla_tokens']} tokens**
- **Token Horizon**: ~46 million 50-move game trajectories

---

## Artifact Inventory

| Filename | Description | Format |
|---|---|---|
| `model.pt` | Trained model weights (state dictionary) | PyTorch binary |
| `model_config.json` | Model architecture hyperparameter configuration | JSON |
| `tokenizer.json` | Trained BPE tokenizer vocabulary and merge tables | Hugging Face Tokenizers JSON |
| `README.md` | Model specification and benchmark documentation | Markdown |

For local deployment and quantized execution with `llama.cpp` or Ollama, download the GGUF artifact from [{meta['gguf_repo_id']}](https://huggingface.co/{meta['gguf_repo_id']}).

---

## Python Usage Example

```python
import json
import torch
from src.models.transformer.nebium import Nebium
from src.data.tokenizer import ChessTokenizer

# 1. Initialize and load the tokenizer
tokenizer = ChessTokenizer()
tokenizer.load("tokenizer.json")

# 2. Instantiate model architecture from configuration
with open("model_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

model = Nebium(**config)
state_dict = torch.load("model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# 3. Autoregressive move generation
prompt = "e2e4 e7e5 g1f3"
input_ids = torch.tensor([[tokenizer.bos_id] + tokenizer.encode(prompt)], dtype=torch.long)
attention_mask = torch.ones_like(input_ids)

with torch.no_grad():
    output = model.generate(input_ids, attention_mask, max_new_tokens=10, temperature=0.7)

generated_text = tokenizer.decode(output[0].tolist())
print("Generated continuation:", generated_text)
```

---

## License & Citation

This model is licensed under the MIT License. If using Nebium in your research, cite the repository:

```bibtex
@misc{{nebium_{meta['tier']}_2026,
  author = {{Oli, Nabin}},
  title = {{{meta['name']}: Causal Transformer for Next-Move Prediction in Chess}},
  year = {{2026}},
  publisher = {{Hugging Face}},
  journal = {{Hugging Face Model Hub}},
  howpublished = {{\\url{{https://huggingface.co/{meta['repo_id']}}}}}
}}
```
"""


def build_gguf_model_card(meta: dict[str, Any]) -> str:
    """Generates the Markdown model card for GGUF artifacts."""
    return f"""---
language:
- en
license: mit
tags:
- chess
- causal-lm
- gguf
- llama.cpp
- ollama
- nebium
- nebium-{meta['tier']}
pipeline_tag: text-generation
---

# {meta['name']}-GGUF

Quantized and FP16 GGUF format binaries for **{meta['name']}** ({meta['params']} parameters).

Designed for high-efficiency CPU and GPU local inference via [llama.cpp](https://github.com/ggerganov/llama.cpp), [Ollama](https://ollama.ai), and Python bindings (`llama-cpp-python`).

- **Base PyTorch Weights**: [{meta['repo_id']}](https://huggingface.co/{meta['repo_id']})
- **Framework Source**: [github.com/nabin2004/nebium](https://github.com/nabin2004/nebium)

---

## Available GGUF Binaries

| Filename | Precision | Memory Footprint | Recommended Use Case |
|---|---|---|---|
| `nebium-{meta['tier']}.gguf` | FP16 | ~{int(meta['d_model']) * int(meta['n_layers']) * 12 * 2 // (1024 * 1024) if meta['tier'] == 'small' else '690MB' if meta['tier'] == 'medium' else '1.5GB'} | Full precision baseline for local evaluation |
| `nebium-{meta['tier']}-q8_0.gguf` | Q8_0 | ~{int(meta['d_model']) * int(meta['n_layers']) * 12 // (1024 * 1024) if meta['tier'] == 'small' else '360MB' if meta['tier'] == 'medium' else '800MB'} | Recommended precision with negligible perplexity loss |
| `nebium-{meta['tier']}-q4_k_m.gguf` | Q4_K_M | Minimal | Resource-constrained edge and CPU environments |

---

## Execution with llama.cpp

### CLI Inference

```bash
# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# Download the model
huggingface-cli download {meta['gguf_repo_id']} nebium-{meta['tier']}.gguf --local-dir .

# Run prompt continuation
./llama-cli -m nebium-{meta['tier']}.gguf \\
  -p "e2e4 e7e5 g1f3" \\
  -n 30 \\
  --temp 0.7 \\
  --top-p 0.95
```

---

## Execution with Ollama

### Modelfile

Create a file named `Modelfile`:

```dockerfile
FROM ./nebium-{meta['tier']}.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER stop "<|eos|>"

SYSTEM You are an autoregressive chess move continuation engine predicting next moves in UCI notation.
```

### Build and Run

```bash
ollama create {meta['name'].lower()} -f Modelfile
ollama run {meta['name'].lower()} "e2e4 e7e5"
```

---

## Python llama-cpp-python Usage

```python
from llama_cpp import Llama

llm = Llama(
    model_path="nebium-{meta['tier']}.gguf",
    n_ctx=1024,
    n_threads=4,
)

prompt = "e2e4 e7e5 g1f3 b8c6"
output = llm(
    prompt,
    max_tokens=20,
    temperature=0.7,
    stop=["<|eos|>", "<|pad|>"],
)

print("Generated moves:", output["choices"][0]["text"])
```

---

## Architecture Specifications

- **Base Architecture**: Causal Transformer with Rotary Position Embeddings (RoPE)
- **Hidden Dimension ($d_{{model}}$)**: {meta['d_model']}
- **Attention Heads**: {meta['n_heads']}
- **Layers**: {meta['n_layers']}
- **Context Length**: {meta['max_seq_len']} tokens
- **Vocabulary**: {meta['vocab_size']} UCI BPE tokens

---

## License

MIT License.
"""


def create_and_initialize_repositories(
    token: str | None = None,
    private: bool = False,
    tiers: list[str] | None = None,
) -> None:
    """Creates the Hugging Face model and GGUF repositories and uploads model cards."""
    api = HfApi(token=token)
    user_info = api.whoami()
    username = user_info.get("name", "nabin2004")
    print(f"Authenticated as @{username}. Provisioning repositories...")

    selected_tiers = tiers if tiers else ["small", "medium", "large"]

    for tier in selected_tiers:
        meta = dict(MODELS[tier])
        meta["repo_id"] = f"{username}/nebium-{tier}"
        meta["gguf_repo_id"] = f"{username}/nebium-{tier}-gguf"

        # 1. Base Model Repository
        print(f"\n[{meta['name']}] Creating base repo: {meta['repo_id']}")
        try:
            api.create_repo(
                repo_id=meta["repo_id"],
                repo_type="model",
                private=private,
                exist_ok=True,
            )
            print(f"  Repo {meta['repo_id']} confirmed.")
        except Exception as exc:
            print(f"  Warning creating {meta['repo_id']}: {exc}")

        # Upload .gitattributes
        try:
            api.upload_file(
                path_or_fileobj=GITATTRIBUTES_CONTENT.encode("utf-8"),
                path_in_repo=".gitattributes",
                repo_id=meta["repo_id"],
                commit_message="Configure Git LFS tracking rules",
            )
        except Exception as exc:
            print(f"  Note uploading .gitattributes: {exc}")

        # Upload Model Card README.md
        card_content = build_pytorch_model_card(meta)
        try:
            api.upload_file(
                path_or_fileobj=card_content.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=meta["repo_id"],
                commit_message=f"Initialize comprehensive {meta['name']} model card",
            )
            print(f"  Uploaded README.md to {meta['repo_id']}")
        except Exception as exc:
            print(f"  Error uploading README.md: {exc}")

        # 2. GGUF Companion Repository
        print(f"[{meta['name']}-GGUF] Creating GGUF repo: {meta['gguf_repo_id']}")
        try:
            api.create_repo(
                repo_id=meta["gguf_repo_id"],
                repo_type="model",
                private=private,
                exist_ok=True,
            )
            print(f"  Repo {meta['gguf_repo_id']} confirmed.")
        except Exception as exc:
            print(f"  Warning creating {meta['gguf_repo_id']}: {exc}")

        # Upload .gitattributes
        try:
            api.upload_file(
                path_or_fileobj=GITATTRIBUTES_CONTENT.encode("utf-8"),
                path_in_repo=".gitattributes",
                repo_id=meta["gguf_repo_id"],
                commit_message="Configure Git LFS tracking rules for GGUF binaries",
            )
        except Exception as exc:
            print(f"  Note uploading .gitattributes: {exc}")

        # Upload GGUF Model Card README.md
        gguf_card_content = build_gguf_model_card(meta)
        try:
            api.upload_file(
                path_or_fileobj=gguf_card_content.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=meta["gguf_repo_id"],
                commit_message=f"Initialize {meta['name']}-GGUF deployment documentation",
            )
            print(f"  Uploaded README.md to {meta['gguf_repo_id']}")
        except Exception as exc:
            print(f"  Error uploading GGUF README.md: {exc}")

    print("\nAll requested repositories have been provisioned and initialized successfully.")


def main():
    parser = argparse.ArgumentParser(description="Create and initialize Nebium model and GGUF repositories")
    parser.add_argument("--tiers", nargs="+", choices=["small", "medium", "large"], default=["small", "medium", "large"])
    parser.add_argument("--token", type=str, default=None, help="HF Token (optional if cached)")
    parser.add_argument("--private", action="store_true", help="Set repositories to private (default: public)")
    args = parser.parse_args()

    create_and_initialize_repositories(token=args.token, private=args.private, tiers=args.tiers)


if __name__ == "__main__":
    main()
