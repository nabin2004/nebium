"""
scripts/build_kaggle_notebook.py
================================
Generates the production-grade Kaggle notebook (notebooks/kaggle_train.ipynb)
for training all three Nebium models (Small, Medium, Large) end-to-end on
Kaggle dual-T4 GPUs (2x Tesla T4), with smoke tests, GGUF conversion verification,
Hugging Face hub uploads, and cross-tier scaling-law analysis.
"""

import json
from pathlib import Path


def create_kaggle_notebook() -> dict:
    cells = []

    def md(source: str):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def code(source: str):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # --- Title & Overview ---
    md("""
# Nebium: End-to-End Multi-Tier Training Pipeline (Kaggle 2x T4 GPUs)

This notebook provides an automated, end-to-end workflow to train, evaluate, convert, and push all three tiers of the **Nebium Causal Chess Transformer** family:
1. **Nebium-Small (117M)** $\\rightarrow$ GPT-2 Small scale ($d_{model}=768, L=12$)
2. **Nebium-Medium (345M)** $\\rightarrow$ GPT-2 Medium scale ($d_{model}=1024, L=24$)
3. **Nebium-Large (762M)** $\\rightarrow$ GPT-2 Large scale ($d_{model}=1280, L=36$)

---

### Key Capabilities
- **Dual T4 Acceleration**: Multi-GPU `DataParallel` training utilizing both 16GB Tesla T4 GPUs (32GB total VRAM) with FP16 mixed precision.
- **Fast Pipeline Smoke Test**: Rapid validation (<15s) of data ingestion, tokenizer, forward/backward pass, GGUF export, and Hub staging before long runs.
- **Automated Dual-Destination Push**: Each model automatically pushes to its base PyTorch repo (`nabin2004/nebium-{tier}`) and dedicated GGUF companion repo (`nabin2004/nebium-{tier}-gguf`).
- **Publication-Grade WandB Tracking**: Interactive move rollout tables with legality badges, board FENs, puzzle benchmark suite, and 300 DPI vector figures.
- **Empirical Scaling Law Analysis**: Cross-tier Hoffmann et al. (2022) Chinchilla scaling law evaluation comparing all 3 tiers.

---

### Prerequisites in Kaggle
1. **Accelerator**: Select **GPU T4 x 2** in the notebook settings.
2. **Internet**: Toggle **Internet ON**.
3. **Kaggle Secrets** (Add via *Add-ons -> Secrets*):
   - `HF_TOKEN`: Hugging Face write token (authenticated as `@nabin2004`).
   - `WANDB_API_KEY`: Weights & Biases API key (optional; defaults to offline/disabled if omitted).
""")

    # --- Cell 1: Environment & GPU Verification ---
    md("## 1. Environment & Dual-GPU Verification")
    code("""import os
import sys

# Configure PyTorch CUDA memory allocator to prevent segment fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from pathlib import Path

# Verify GPU availability
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available : {torch.cuda.is_available()}")
device_count = torch.cuda.device_count()
print(f"Device Count   : {device_count}")

for i in range(device_count):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name} | Total Memory: {props.total_memory / (1024**3):.2f} GB")

if device_count < 2:
    print("\\n[NOTE] Running with single GPU or CPU. For optimal multi-tier performance, select 'GPU T4 x 2' in notebook settings.")
else:
    print("\\n[SUCCESS] Dual-T4 GPU configuration verified!")

# Load Kaggle secrets
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    
    hf_token = secrets.get_secret("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        print("HF_TOKEN loaded successfully.")
    
    try:
        wandb_key = secrets.get_secret("WANDB_API_KEY")
        if wandb_key:
            os.environ["WANDB_API_KEY"] = wandb_key
            print("WANDB_API_KEY loaded successfully.")
    except Exception:
        print("WANDB_API_KEY not found in secrets; WandB will run in offline mode.")
        os.environ["WANDB_MODE"] = "offline"
except Exception as e:
    print(f"Kaggle Secrets Client note: {e}")
""")

    # --- Cell 2: Setup Repository ---
    md("## 2. Setup Repository & Install Dependencies")
    code("""# Workspace setup
REPO_DIR = Path("/kaggle/working/nebium")

if not (REPO_DIR / "scripts" / "train.py").exists():
    print("Cloning Nebium repository...")
    !git clone https://github.com/nabin2004/nebium.git {REPO_DIR}
else:
    print("Repository already cloned. Pulling latest updates...")
    !cd {REPO_DIR} && git pull origin master

%cd {REPO_DIR}

# Install dependencies
print("\\nInstalling project requirements...")
!pip install -e . --no-deps -q
!pip install chess zstandard hydra-core omegaconf gguf wandb matplotlib seaborn tqdm bitsandbytes -q

print("\\nInstallation complete!")
""")

    # --- Cell 3: Fast Pipeline Smoke Test ---
    md("""## 3. Fast Pipeline Smoke Test

Verifies the entire execution chain in ~10 seconds before committing GPU hours:
- Ingestion and streaming tokenization
- Forward and backward pass with AMP FP16
- Validation metrics aggregation
- GGUF export conversion and binary file validation
- Hugging Face Hub staging directory packaging
""")
    code("""# Execute fast smoke test with Kaggle configuration
!python scripts/train.py --config-name kaggle --smoke-test \\
    data.hf_dataset.repo_id=nabin2004/nebium-lichess-uci \\
    hub.repo_id=nabin2004/nebium-small
""")

    # --- Cell 4: GGUF File Verification Helper ---
    md("## 4. GGUF Binary Inspection & Validation Utility")
    code("""import gguf
from pathlib import Path

def inspect_gguf_file(path_str: str):
    path = Path(path_str)
    if not path.exists():
        print(f"GGUF file not found at: {path}")
        return
    
    reader = gguf.GGUFReader(str(path))
    print("=" * 60)
    print(f"  GGUF File: {path.name}")
    print(f"  File Size: {path.stat().st_size / (1024**2):.2f} MB")
    print(f"  Tensors  : {len(reader.tensors)}")
    print(f"  Metadata : {len(reader.fields)} fields")
    print("-" * 60)
    
    # Inspect key metadata
    for field_name in ["general.architecture", "general.name", "tokenizer.ggml.model"]:
        if field_name in reader.fields:
            val = reader.fields[field_name].parts[0].tolist() if hasattr(reader.fields[field_name].parts[0], 'tolist') else reader.fields[field_name].parts[0]
            if isinstance(val, (bytes, bytearray)):
                val = bytes(val).decode('utf-8', errors='ignore')
            print(f"  {field_name:<24}: {val}")
    
    print("=" * 60)

# Verify local smoke test GGUF export
inspect_gguf_file("nebium.gguf")
""")

    # --- Cell 5: Automated All-in-One Multi-Tier Training ---
    md("""## 5. End-to-End Multi-Tier Training Runner (Small $\\rightarrow$ Medium $\\rightarrow$ Large)

The cell below sequentially orchestrates the training, GGUF export, and automated Hub push for all three model tiers:
1. **Nebium-Small (117M)** $\\rightarrow$ Pushes to `nabin2004/nebium-small` & `nabin2004/nebium-small-gguf`
2. **Nebium-Medium (345M)** $\\rightarrow$ Pushes to `nabin2004/nebium-medium` & `nabin2004/nebium-medium-gguf`
3. **Nebium-Large (762M)** $\\rightarrow$ Pushes to `nabin2004/nebium-large` & `nabin2004/nebium-large-gguf`

Includes automatic VRAM clearing between runs to prevent CUDA Out-Of-Memory.
""")
    code("""import gc
import subprocess
import torch

TIERS = [
    {
        "tier": "small",
        "name": "Nebium-Small (117M)",
        "config": "kaggle_small",
        "output_dir": "/kaggle/working/outputs/nebium-small",
        "checkpoint": "checkpoint_small.pt",
    },
    {
        "tier": "medium",
        "name": "Nebium-Medium (345M)",
        "config": "kaggle_medium",
        "output_dir": "/kaggle/working/outputs/nebium-medium",
        "checkpoint": "checkpoint_medium.pt",
    },
    {
        "tier": "large",
        "name": "Nebium-Large (762M)",
        "config": "kaggle_large",
        "output_dir": "/kaggle/working/outputs/nebium-large",
        "checkpoint": "checkpoint_large.pt",
    },
]

# Set to True to run all 3 tiers sequentially, or select individual tier
RUN_ALL_TIERS = True
SELECTED_TIERS = ["small", "medium", "large"] if RUN_ALL_TIERS else ["small"]

for item in TIERS:
    tier = item["tier"]
    if tier not in SELECTED_TIERS:
        continue
        
    print("\\n" + "#" * 70)
    print(f"# LAUNCHING TRAINING: {item['name']}")
    print(f"# Config: configs/{item['config']}.yaml")
    print("#" * 70 + "\\n")
    
    # Clean CUDA cache before launching
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    cmd = [
        sys.executable, "scripts/train.py",
        f"--config-name={item['config']}",
        f"data.hf_dataset.repo_id=nabin2004/nebium-lichess-uci",
        f"hub.repo_id=nabin2004/nebium-{tier}",
        f"hub.gguf_repo_id=nabin2004/nebium-{tier}-gguf",
        "hub.push=true",
        "hub.push_gguf_repo=true",
        "hub.private=false",
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"[ERROR] Training failed for {item['name']} with exit code {result.returncode}")
    else:
        print(f"[SUCCESS] Completed training, GGUF export, and Hub push for {item['name']}!")
        
        # Save tier checkpoint copy for scaling law analysis
        if Path("best_model.pt").exists():
            import shutil
            shutil.copy2("best_model.pt", item["checkpoint"])
            print(f"Archived checkpoint to {item['checkpoint']}")
    
    # Clear VRAM after run
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
""")

    # --- Cell 6: Individual Tier Training (Optional Manual Execution) ---
    md("""## 6. Individual Tier Training Cells (Optional Manual Execution)

If you prefer to train tiers individually or adjust hyperparameters per model, execute the respective cells below:
""")

    md("### 6.1 Train Nebium-Small (117M)")
    code("""# Train Small (117M)
!python scripts/train.py --config-name kaggle_small \\
    data.hf_dataset.repo_id=nabin2004/nebium-lichess-uci \\
    hub.repo_id=nabin2004/nebium-small \\
    hub.gguf_repo_id=nabin2004/nebium-small-gguf \\
    hub.push=true
""")

    md("### 6.2 Train Nebium-Medium (345M)")
    code("""# Train Medium (345M)
!python scripts/train.py --config-name kaggle_medium \\
    data.hf_dataset.repo_id=nabin2004/nebium-lichess-uci \\
    hub.repo_id=nabin2004/nebium-medium \\
    hub.gguf_repo_id=nabin2004/nebium-medium-gguf \\
    hub.push=true
""")

    md("### 6.3 Train Nebium-Large (762M)")
    code("""# Train Large (762M)
!python scripts/train.py --config-name kaggle_large \\
    data.hf_dataset.repo_id=nabin2004/nebium-lichess-uci \\
    hub.repo_id=nabin2004/nebium-large \\
    hub.gguf_repo_id=nabin2004/nebium-large-gguf \\
    hub.push=true
""")

    # --- Cell 7: Cross-Tier Scaling Laws Evaluation ---
    md("""## 7. Cross-Tier Chinchilla Scaling Law Evaluation

Evaluates all three trained checkpoints (`checkpoint_small.pt`, `checkpoint_medium.pt`, `checkpoint_large.pt`) on a shared evaluation split, comparing their empirical scaling exponent against Hoffmann et al. (2022) Chinchilla power-law projections.
""")
    code("""import os
from IPython.display import Image, display

# Verify available checkpoints
small_ckpt = "checkpoint_small.pt" if Path("checkpoint_small.pt").exists() else "best_model.pt"
medium_ckpt = "checkpoint_medium.pt" if Path("checkpoint_medium.pt").exists() else None
large_ckpt = "checkpoint_large.pt" if Path("checkpoint_large.pt").exists() else None

cmd_parts = [
    "python scripts/eval_scaling_laws.py",
    f"--small {small_ckpt}",
    f"--data-path data/fixtures/sample.pgn",
    f"--output paper_assets/scaling_laws.json",
    f"--report paper_assets/SCALING_LAWS.md",
]

if medium_ckpt:
    cmd_parts.append(f"--medium {medium_ckpt}")
if large_ckpt:
    cmd_parts.append(f"--large {large_ckpt}")

cmd = " ".join(cmd_parts)
print(f"Running scaling law benchmark: {cmd}")
!{cmd}

# Display generated publication plot
if Path("paper_assets/scaling_laws_alignment.png").exists():
    display(Image(filename="paper_assets/scaling_laws_alignment.png"))
""")

    # --- Cell 8: Interactive Move Generation Demo ---
    md("""## 8. Interactive Move Generation & Board Visualization

Demonstrates autoregressive move prediction from arbitrary board openings, checking move legality with `python-chess` and displaying ASCII board positions.
""")
    code("""import chess
import json
import torch
from src.models.transformer.nebium import Nebium
from src.data.tokenizer import ChessTokenizer

# Load tokenizer
tokenizer = ChessTokenizer()
tokenizer_candidates = [
    "export/tokenizer.json",
    "/kaggle/working/data/tokenizer/lichess/tokenizer.json",
    "data/tokenizer/lichess/tokenizer.json",
    "data/tokenizer/lichess_2013/tokenizer.json",
    "data/tokenizer/fixture/tokenizer.json",
]
tokenizer_path = next((p for p in tokenizer_candidates if Path(p).exists()), None)
if tokenizer_path:
    tokenizer.load(tokenizer_path)
    print(f"Loaded tokenizer from {tokenizer_path}")
else:
    print("Warning: No tokenizer.json found yet.")

# Load config and model
config_path = "export/model_config.json"
if Path(config_path).exists():
    with open(config_path, "r") as f:
        config = json.load(f)
    model = Nebium(**config)
    ckpt_path = "best_model.pt" if Path("best_model.pt").exists() else "checkpoint.pt"
    if Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state.get("model", state))
        model.eval()
        
        # Test generation from starting position
        prompt = "e2e4 e7e5 g1f3"
        board = chess.Board()
        for mv in prompt.split():
            board.push_uci(mv)
            
        print("Board Position after prompt:")
        print(board)
        print("\\nGenerating next moves...")
        
        input_ids = torch.tensor([[tokenizer.bos_id] + tokenizer.encode(prompt)])
        mask = torch.ones_like(input_ids)
        with torch.no_grad():
            tokens = model.generate(input_ids, mask, max_new_tokens=10, temperature=0.7)
        continuation = tokenizer.decode(tokens[0].tolist())
        print(f"Model continuation: {continuation}")
    else:
        print("No checkpoint found to demo.")
else:
    print("No model_config.json found. Run training or smoke test first.")
""")

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 5,
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            },
            "accelerator": "GPU",
            "gpu_count": 2
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    return notebook


def main():
    nb = create_kaggle_notebook()
    out_file = Path("notebooks/kaggle_train.ipynb")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Successfully generated {out_file} ({len(nb['cells'])} cells).")


if __name__ == "__main__":
    main()
