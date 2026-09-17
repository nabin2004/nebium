import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


REQUIRED_FILES = ("moves.txt", "manifest.json", "tokenizer.json")


def _dataset_card(repo_id: str, n_sequences: int) -> str:
    return (
        "---\n"
        "tags:\n"
        "- chess\n"
        "- nebium\n"
        "---\n\n"
        f"# {repo_id}\n\n"
        "Processed Lichess games as space-separated UCI move sequences "
        f"(`moves.txt`, {n_sequences} games) plus `tokenizer.json` and `manifest.json`.\n"
        "Built by Nebium so later runs can skip PGN parsing.\n"
    )


def pull_hf_dataset(
    repo_id: str,
    revision: str,
    processed_dir: Path,
    tokenizer_dir: Path,
) -> bool:
    try:
        cache = Path(
            snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision)
        )
    except Exception:
        return False
    if not all((cache / name).exists() for name in REQUIRED_FILES):
        return False
    processed_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cache / "moves.txt", processed_dir / "moves.txt")
    shutil.copy2(cache / "manifest.json", processed_dir / "manifest.json")
    shutil.copy2(cache / "tokenizer.json", tokenizer_dir / "tokenizer.json")
    return True


def push_hf_dataset(
    repo_id: str,
    processed_dir: Path,
    tokenizer_dir: Path,
    n_sequences: int,
    private: bool = True,
) -> str:
    for name in ("moves.txt", "manifest.json"):
        src = processed_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing {src} for Hub dataset upload")
    
    tokenizer = tokenizer_dir / "tokenizer.json"
    if not tokenizer.exists():
        raise FileNotFoundError(f"Missing {tokenizer} for Hub dataset upload")

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    
    # Upload files directly to avoid copying large moves.txt locally
    api.upload_file(
        path_or_fileobj=str(processed_dir / "moves.txt"),
        path_in_repo="moves.txt",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add processed Nebium UCI corpus (moves.txt)",
    )
    api.upload_file(
        path_or_fileobj=str(processed_dir / "manifest.json"),
        path_in_repo="manifest.json",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add manifest.json",
    )
    api.upload_file(
        path_or_fileobj=str(tokenizer),
        path_in_repo="tokenizer.json",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add tokenizer.json",
    )
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md', encoding="utf-8") as tf:
        tf.write(_dataset_card(repo_id, n_sequences))
        tf_path = tf.name
        
    api.upload_file(
        path_or_fileobj=tf_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add README.md",
    )
    Path(tf_path).unlink(missing_ok=True)
    
    return repo_id
