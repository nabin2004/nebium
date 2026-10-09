"""
scripts/push_kaggle_to_hf.py
============================
Standalone script to push Nebium models, GGUF binaries, and tokenizers
directly from a Kaggle dataset (e.g. 'nabinoli2004/outputt') to Hugging Face Hub.

Can be run directly inside a Kaggle notebook cell or as a standalone script.
"""

import os
import sys
import glob
import json
import shutil
from pathlib import Path

# Install / verify huggingface_hub
try:
    from huggingface_hub import HfApi, create_repo, login
except ImportError:
    import subprocess
    print("Installing huggingface_hub...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
    from huggingface_hub import HfApi, create_repo, login


def find_dataset_root(search_hints=None) -> Path:
    """Locate the root directory of the Kaggle dataset."""
    if search_hints is None:
        search_hints = [
            "/kaggle/input/outputt",
            "/kaggle/input/outputt/nebium",
            "/kaggle/input/*outputt*",
            "./",
        ]
    
    for hint in search_hints:
        matches = glob.glob(hint)
        for m in matches:
            p = Path(m)
            # Look for best_model.pt or nebium.gguf
            if (p / "best_model.pt").exists() or (p / "nebium.gguf").exists():
                print(f"[+] Found dataset root at: {p}")
                return p
            # Check subdirectories
            for sub in p.iterdir():
                if sub.is_dir() and ((sub / "best_model.pt").exists() or (sub / "nebium.gguf").exists()):
                    print(f"[+] Found dataset root at: {sub}")
                    return sub

    # Fallback search across /kaggle/input
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for pt in kaggle_input.glob("**/best_model.pt"):
            print(f"[+] Found checkpoint at: {pt.parent}")
            return pt.parent

    raise FileNotFoundError("Could not locate dataset containing best_model.pt or nebium.gguf")


def extract_clean_fp16_weights(raw_checkpoint_path: Path, output_path: Path):
    """
    Extracts purely the model state_dict in FP16, stripping bulky optimizer
    states (which bloat the 762M file to 8.5GB).
    Reduces upload size to ~1.5 GB.
    """
    import torch
    print(f"[*] Loading raw checkpoint to extract clean FP16 weights: {raw_checkpoint_path}...")
    checkpoint = torch.load(raw_checkpoint_path, map_location="cpu", weights_only=False)
    
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Convert tensors to FP16 to save bandwidth and memory
    fp16_state_dict = {
        k: v.half() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
        for k, v in state_dict.items()
    }

    print(f"[*] Saving clean FP16 weights to: {output_path}...")
    torch.save(fp16_state_dict, output_path)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[+] Clean FP16 model saved ({size_mb:.1f} MB vs raw {raw_checkpoint_path.stat().st_size / (1024**3):.2f} GB).")


def build_large_model_card() -> str:
    """Generate publication model card for Nebium-Large."""
    return """---
language:
- en
license: mit
tags:
- chess
- causal-transformer
- rope
- swiglu
- rmsnorm
- reinforcement-learning
pipeline_tag: text-generation
---

# Nebium-Large (762M) — Autoregressive Chess Transformer

Nebium is a decoder-only causal transformer trained to predict the next chess move purely from coordinate Universal Chess Interface (UCI) token sequences.

## Architectural Highlights
- **Parameter Count:** 762 Million ($L=36$, $d_{\\text{model}}=1280$, $H=20$)
- **Positional Encoding:** Rotary Position Embeddings (RoPE)
- **Non-Linearity:** SwiGLU Gating ($d_{ff} = 3,424$)
- **Normalization:** Pre-RMSNorm
- **Vocabulary:** 2,056-token domain Byte-Pair Encoding (BPE) (exhausts all 1,968 observed UCI moves + 83 sub-units + 5 special tokens)
- **Training Infrastructure:** Dual NVIDIA T4 GPUs via PyTorch DDP with FP16 AMP and 8-bit AdamW

## Repository Contents
- `pytorch_model.bin`: Clean FP16 model weights (~1.5 GB)
- `best_model_full.pt`: Full training checkpoint including optimizer states (optional)
- `tokenizer.json`: Domain BPE tokenizer vocabulary and merge rules
- `config.yaml`: Hydra training and architecture configuration

## Academic Reference
Assessment 2 Research Report, School of Computing and Digital Technology, Birmingham City University.
"""


def main():
    # 1. Obtain Hugging Face Token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        try:
            # Check Kaggle User Secrets
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            hf_token = user_secrets.get_secret("HF_TOKEN")
        except Exception:
            pass

    if not hf_token:
        import getpass
        print("[!] HF_TOKEN not found in environment or Kaggle Secrets.")
        hf_token = getpass.getpass("Enter your Hugging Face Write Token: ").strip()

    if not hf_token:
        print("[ERROR] Hugging Face token is required to push models.")
        sys.exit(1)

    login(token=hf_token)
    api = HfApi(token=hf_token)
    user_info = api.whoami()
    username = user_info["name"]
    print(f"[+] Authenticated as Hugging Face user: {username}")

    # 2. Locate Kaggle Dataset
    dataset_root = find_dataset_root()

    # 3. Setup Staging Directory
    staging_dir = Path("/kaggle/working/hf_staging") if Path("/kaggle/working").exists() else Path("./hf_staging")
    staging_dir.mkdir(parents=True, exist_ok=True)

    # 4. Prepare PyTorch Model Repository
    model_repo_id = f"{username}/nebium-large"
    gguf_repo_id = f"{username}/nebium-large-gguf"

    print(f"\n[+] Target Model Repo: https://huggingface.co/{model_repo_id}")
    print(f"[+] Target GGUF Repo:  https://huggingface.co/{gguf_repo_id}")

    api.create_repo(repo_id=model_repo_id, repo_type="model", exist_ok=True)
    api.create_repo(repo_id=gguf_repo_id, repo_type="model", exist_ok=True)

    # Gitattributes for Git LFS
    gitattributes = "*.pt filter=lfs diff=lfs merge=lfs -text\n*.bin filter=lfs diff=lfs merge=lfs -text\n*.gguf filter=lfs diff=lfs merge=lfs -text\n"
    api.upload_file(
        path_or_fileobj=gitattributes.encode("utf-8"),
        path_in_repo=".gitattributes",
        repo_id=model_repo_id,
    )
    api.upload_file(
        path_or_fileobj=gitattributes.encode("utf-8"),
        path_in_repo=".gitattributes",
        repo_id=gguf_repo_id,
    )

    # 5. Extract and upload clean FP16 weights
    raw_pt = dataset_root / "best_model.pt"
    if not raw_pt.exists():
        raw_pt = dataset_root / "checkpoint.pt"

    if raw_pt.exists():
        fp16_bin = staging_dir / "pytorch_model.bin"
        try:
            extract_clean_fp16_weights(raw_pt, fp16_bin)
            print(f"[+] Uploading clean FP16 weights to {model_repo_id}...")
            api.upload_file(
                path_or_fileobj=str(fp16_bin),
                path_in_repo="pytorch_model.bin",
                repo_id=model_repo_id,
            )
        except Exception as e:
            print(f"[!] Could not extract FP16 directly ({e}). Uploading raw checkpoint...")
            api.upload_file(
                path_or_fileobj=str(raw_pt),
                path_in_repo="best_model.pt",
                repo_id=model_repo_id,
            )

    # 6. Upload Tokenizer
    tok_candidates = list(dataset_root.glob("**/tokenizer.json"))
    if tok_candidates:
        tok_file = tok_candidates[0]
        print(f"[+] Uploading tokenizer ({tok_file}) to {model_repo_id}...")
        api.upload_file(
            path_or_fileobj=str(tok_file),
            path_in_repo="tokenizer.json",
            repo_id=model_repo_id,
        )

    # 7. Upload Config / Logs
    config_candidates = list(dataset_root.glob("**/config.yaml"))
    if config_candidates:
        print(f"[+] Uploading config ({config_candidates[0]}) to {model_repo_id}...")
        api.upload_file(
            path_or_fileobj=str(config_candidates[0]),
            path_in_repo="config.yaml",
            repo_id=model_repo_id,
        )

    # Model Card
    api.upload_file(
        path_or_fileobj=build_large_model_card().encode("utf-8"),
        path_in_repo="README.md",
        repo_id=model_repo_id,
    )

    # 8. Upload GGUF binary
    gguf_candidates = list(dataset_root.glob("**/*.gguf"))
    if gguf_candidates:
        gguf_file = gguf_candidates[0]
        print(f"[+] Uploading GGUF binary ({gguf_file}, {gguf_file.stat().st_size / (1024**3):.2f} GB) to {gguf_repo_id}...")
        api.upload_file(
            path_or_fileobj=str(gguf_file),
            path_in_repo=gguf_file.name,
            repo_id=gguf_repo_id,
        )
        # GGUF Model card
        gguf_card = f"""---
license: mit
tags:
- llama.cpp
- gguf
- chess
---
# {gguf_file.name} (Nebium-Large 4-bit/FP16 GGUF)

Edge-deployable quantized binary for CPU inference using `llama.cpp` or Ollama.
"""
        api.upload_file(
            path_or_fileobj=gguf_card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=gguf_repo_id,
        )

    # Cleanup staging
    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)

    print("\n========================================================")
    print(" [✓] Upload Complete!")
    print(f" PyTorch Model: https://huggingface.co/{model_repo_id}")
    print(f" GGUF Model:    https://huggingface.co/{gguf_repo_id}")
    print("========================================================\n")


if __name__ == "__main__":
    main()
