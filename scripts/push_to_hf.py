"""
scripts/push_to_hf.py
=====================
CLI utility to upload Nebium model checkpoints, configuration, tokenizer, and
GGUF binaries to Hugging Face Hub repositories across Small, Medium, and Large tiers.
"""

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

from scripts.create_hf_repos import MODELS, build_pytorch_model_card, build_gguf_model_card, GITATTRIBUTES_CONTENT


def push_tier(
    tier: str,
    api: HfApi,
    username: str,
    checkpoint_path: str = "best_model.pt",
    tokenizer_path: str | None = None,
    gguf_path: str | None = None,
    push_gguf: bool = True,
    token: str | None = None,
) -> None:
    meta = dict(MODELS[tier])
    repo_id = f"{username}/nebium-{tier}"
    gguf_repo_id = f"{username}/nebium-{tier}-gguf"

    print(f"\n==================================================")
    print(f"  Pushing {meta['name']} to Hugging Face")
    print(f"  Target PyTorch Repo: {repo_id}")
    print(f"  Target GGUF Repo:    {gguf_repo_id}")
    print(f"==================================================")

    # 1. Base PyTorch Repo
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)
    except Exception as exc:
        print(f"Repo check warning: {exc}")

    # Resolve tokenizer path
    if tokenizer_path is None or not os.path.exists(tokenizer_path):
        candidate_tokenizers = [
            "tokenizer.json",
            "data/tokenizer/lichess_2013/tokenizer.json",
            "data/tokenizer/fixture/tokenizer.json",
            "export/tokenizer.json",
        ]
        tokenizer_path = next((p for p in candidate_tokenizers if os.path.exists(p)), None)

    # Upload PyTorch files
    pytorch_files = {}
    if os.path.exists(checkpoint_path):
        pytorch_files[checkpoint_path] = "model.pt"
    if tokenizer_path and os.path.exists(tokenizer_path):
        pytorch_files[tokenizer_path] = "tokenizer.json"
    if os.path.exists("model_config.json"):
        pytorch_files["model_config.json"] = "model_config.json"

    # Always ensure .gitattributes and README.md
    api.upload_file(
        path_or_fileobj=GITATTRIBUTES_CONTENT.encode("utf-8"),
        path_in_repo=".gitattributes",
        repo_id=repo_id,
        token=token,
    )
    api.upload_file(
        path_or_fileobj=build_pytorch_model_card(meta).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        token=token,
    )

    for local_f, repo_f in pytorch_files.items():
        print(f"Uploading {local_f} -> {repo_id}:{repo_f}...")
        api.upload_file(
            path_or_fileobj=local_f,
            path_in_repo=repo_f,
            repo_id=repo_id,
            token=token,
        )

    print(f"Successfully uploaded base model to https://huggingface.co/{repo_id}")

    # 2. Companion GGUF Repo
    if push_gguf:
        if gguf_path is None or not os.path.exists(gguf_path):
            candidate_ggufs = [
                f"nebium-{tier}.gguf",
                "nebium.gguf",
                "export/nebium.gguf",
                f"export/gguf/nebium-{tier}.gguf",
            ]
            gguf_path = next((g for g in candidate_ggufs if os.path.exists(g)), None)

        try:
            api.create_repo(repo_id=gguf_repo_id, repo_type="model", exist_ok=True, token=token)
        except Exception as exc:
            print(f"GGUF repo check warning: {exc}")

        api.upload_file(
            path_or_fileobj=GITATTRIBUTES_CONTENT.encode("utf-8"),
            path_in_repo=".gitattributes",
            repo_id=gguf_repo_id,
            token=token,
        )
        api.upload_file(
            path_or_fileobj=build_gguf_model_card(meta).encode("utf-8"),
            path_in_repo="README.md",
            repo_id=gguf_repo_id,
            token=token,
        )

        if tokenizer_path and os.path.exists(tokenizer_path):
            api.upload_file(
                path_or_fileobj=tokenizer_path,
                path_in_repo="tokenizer.json",
                repo_id=gguf_repo_id,
                token=token,
            )

        if gguf_path and os.path.exists(gguf_path):
            print(f"Uploading {gguf_path} -> {gguf_repo_id}:nebium-{tier}.gguf...")
            api.upload_file(
                path_or_fileobj=gguf_path,
                path_in_repo=f"nebium-{tier}.gguf",
                repo_id=gguf_repo_id,
                token=token,
            )
            print(f"Successfully uploaded GGUF model to https://huggingface.co/{gguf_repo_id}")
        else:
            print(f"No local GGUF binary found for {tier}. GGUF model card updated.")


def main():
    parser = argparse.ArgumentParser(description="Push Nebium model checkpoints and GGUF binaries to Hugging Face Hub")
    parser.add_argument("--tier", type=str, choices=["small", "medium", "large", "all"], default="small", help="Model tier")
    parser.add_argument("--checkpoint", type=str, default="best_model.pt", help="Path to checkpoint .pt file")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to tokenizer.json")
    parser.add_argument("--gguf", type=str, default=None, help="Path to .gguf file")
    parser.add_argument("--no-gguf", action="store_true", help="Skip pushing to companion GGUF repo")
    parser.add_argument("--token", type=str, default=None, help="HF Token")
    args = parser.parse_args()

    api = HfApi(token=args.token)
    user_info = api.whoami()
    username = user_info.get("name", "nabin2004")

    tiers = ["small", "medium", "large"] if args.tier == "all" else [args.tier]

    for t in tiers:
        push_tier(
            tier=t,
            api=api,
            username=username,
            checkpoint_path=args.checkpoint,
            tokenizer_path=args.tokenizer,
            gguf_path=args.gguf,
            push_gguf=not args.no_gguf,
            token=args.token,
        )


if __name__ == "__main__":
    main()
