"""
scripts/deploy_space.py
=======================
CLI tool to package and deploy the Nebium Gradio interface to Hugging Face Spaces.
Allows running the full Nebium model family on a public or private Hugging Face Space.

Usage:
    # Deploy to default space nabin2004/nebium-chess:
    uv run python scripts/deploy_space.py

    # Deploy to a custom space name:
    uv run python scripts/deploy_space.py --repo-id nabin2004/nebium-demo

    # Stage files only without uploading:
    uv run python scripts/deploy_space.py --dry-run
"""
try:
    import spaces  # must come before torch
except ImportError:
    spaces = None

import torch

import argparse
import os
import shutil
import sys
from pathlib import Path
from huggingface_hub import HfApi, get_token


ROOT_DIR = Path(__file__).resolve().parent.parent

SPACE_README = """---
title: Nebium Chess Transformer
emoji: ♟️
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Nebium Chess Transformer Family Explorer
---

# ♟️ Nebium Chess Transformer Playground

Interactive visual evaluation playground for the **Nebium** causal Transformer family.
- **Nebium-Small (117M)**
- **Nebium-Medium (345M)**
- **Nebium-Base (6M)**
- **Nebium-Large (762M)**

Trained on Lichess Standard Rated games using Byte-Pair Encoded UCI sequences, RoPE positional embeddings, SwiGLU non-linearities, and RMSNorm.
"""

SPACE_REQUIREMENTS = """gradio>=5.0.0
torch>=2.0.0
tokenizers>=0.15.0
chess>=1.9.0
huggingface-hub>=0.20.0
"""


def stage_space_files(staging_dir: Path) -> None:
    """Prepares clean standalone repository contents for the Hugging Face Space."""
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    # 1. Main app entrypoint
    app_src = ROOT_DIR / "scripts" / "app.py"
    shutil.copy(app_src, staging_dir / "app.py")

    # 2. Source code (models and tokenizers)
    src_dest = staging_dir / "src"
    shutil.copytree(ROOT_DIR / "src", src_dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.zst"))

    # 3. Exported config and tokenizer artifacts for fallback
    export_dest = staging_dir / "export"
    if (ROOT_DIR / "export").exists():
        export_dest.mkdir(parents=True, exist_ok=True)
        for f in ["model_config.json", "tokenizer.json"]:
            src_f = ROOT_DIR / "export" / f
            if src_f.exists():
                shutil.copy(src_f, export_dest / f)

    # 4. Space README with YAML frontmatter
    (staging_dir / "README.md").write_text(SPACE_README, encoding="utf-8")

    # 5. requirements.txt
    (staging_dir / "requirements.txt").write_text(SPACE_REQUIREMENTS, encoding="utf-8")

    print(f"Staged Space files in {staging_dir}:")
    for item in staging_dir.rglob("*"):
        if item.is_file():
            rel = item.relative_to(staging_dir)
            print(f"  - {rel} ({item.stat().st_size:,} bytes)")


def deploy_to_space(
    repo_id: str,
    staging_dir: Path,
    token: str | None = None,
    private: bool = False,
    dry_run: bool = False,
) -> None:
    """Uploads the staged application to Hugging Face Spaces."""
    api = HfApi(token=token)

    if not dry_run:
        print(f"\nAuthenticating with Hugging Face Hub...")
        user_info = api.whoami(token=token)
        username = user_info.get("name")
        print(f"Logged in as: {username}")

        print(f"Creating / verifying Space repository '{repo_id}'...")
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            private=private,
            exist_ok=True,
            token=token,
        )

        print(f"Uploading Space files to '{repo_id}'...")
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(staging_dir),
            repo_type="space",
            token=token,
            commit_message="Deploy Nebium Chess Transformer Gradio interface",
        )

        space_url = f"https://huggingface.co/spaces/{repo_id}"
        print("\n=======================================================")
        print(f"[SUCCESS] Successfully deployed to Hugging Face Spaces!")
        print(f"Live Space URL: {space_url}")
        print("=======================================================\n")
    else:
        print(f"\n[DRY RUN] Staging completed. Skipped upload to {repo_id}.")


def main():
    parser = argparse.ArgumentParser(description="Deploy Nebium Gradio app to Hugging Face Spaces")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="nabin2004/nebium-chess",
        help="Target Hugging Face Space repo ID (e.g. username/space-name)",
    )
    parser.add_argument("--private", action="store_true", help="Make the Space private")
    parser.add_argument("--dry-run", action="store_true", help="Stage files locally without uploading")
    parser.add_argument("--token", type=str, default=None, help="Optional Hugging Face write token")

    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or get_token()
    if not token and not args.dry_run:
        print("Error: No Hugging Face token found. Run `hf auth login` or pass --token / HF_TOKEN.")
        sys.exit(1)

    staging_dir = ROOT_DIR / "scratch" / "space_staging"
    stage_space_files(staging_dir)
    deploy_to_space(
        repo_id=args.repo_id,
        staging_dir=staging_dir,
        token=token,
        private=args.private,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
