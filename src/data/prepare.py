"""
Dataset preparation, caching, and tokenization orchestration for Nebium.

Handles downloading raw Lichess PGN/JSON archives, caching processed UCI move
sequences, maintaining metadata manifests, and training custom BPE tokenizers.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.data.download import ensure_raw_files, url_filename
from src.data.hf_dataset import pull_hf_dataset, push_hf_dataset
from src.data.sources import resolve_path, resolve_raw_files, stream_uci_games
from src.data.tokenizer import ChessTokenizer

MOVES_NAME = "moves.txt"
MANIFEST_NAME = "manifest.json"
TOKENIZER_NAME = "tokenizer.json"



def _urls(cfg: DictConfig) -> list[str]:
    urls = cfg.data.get("urls")
    if not urls:
        return []
    return [str(url) for url in urls]


def _hf_dataset_cfg(cfg: DictConfig) -> dict:
    block = cfg.data.get("hf_dataset")
    if block is None:
        return {"repo_id": None, "push": False, "revision": "main"}
    return OmegaConf.to_container(block, resolve=True)


def _filter_identity(cfg: DictConfig) -> dict:
    return {
        "urls": _urls(cfg),
        "format": cfg.data.format,
        "max_games": int(cfg.data.max_games),
        "min_moves": int(cfg.data.min_moves),
    }


def _source_identity(paths: list[Path]) -> dict:
    return {
        "sources": [str(path.resolve()) for path in paths],
        "source_size": sum(path.stat().st_size for path in paths),
        "source_mtime": max(path.stat().st_mtime for path in paths),
    }


def _expected_manifest(cfg: DictConfig, raw_paths: list[Path] | None = None) -> dict:
    payload = _filter_identity(cfg)
    if raw_paths:
        payload.update(_source_identity(raw_paths))
    return payload


def _manifest_matches(existing: dict, expected: dict) -> bool:
    for key in ("urls", "format", "max_games", "min_moves"):
        if existing.get(key) != expected.get(key):
            return False
    return True


def _read_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_moves(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_corpus(cfg: DictConfig, raw_paths: list[Path], moves_path: Path) -> list[str]:
    moves_path.parent.mkdir(parents=True, exist_ok=True)
    sequences: list[str] = []
    with moves_path.open("w", encoding="utf-8") as out:
        for sequence in stream_uci_games(cfg, raw_paths):
            out.write(sequence + "\n")
            sequences.append(sequence)
    return sequences


def _local_cache_hit(cfg: DictConfig, moves_path: Path, manifest_path: Path) -> bool:
    if bool(cfg.data.get("force_reprocess", False)):
        return False
    existing = _read_manifest(manifest_path)
    return moves_path.exists() and existing is not None and _manifest_matches(existing, _filter_identity(cfg))


def _ensure_raw_paths(cfg: DictConfig, root: str | Path | None) -> list[Path]:
    raw_path = resolve_path(cfg.data.raw_path, root)
    urls = _urls(cfg)
    dest_dir = raw_path if raw_path.suffix == "" or raw_path.is_dir() else raw_path.parent
    preferred: list[Path] = []
    if urls:
        dest_dir.mkdir(parents=True, exist_ok=True)
        preferred = ensure_raw_files(urls, dest_dir)
    return resolve_raw_files(raw_path if raw_path.exists() else dest_dir, cfg.data.format, preferred)


def prepare_corpus(cfg: DictConfig, root: str | Path | None = None) -> tuple[list[str], bool]:
    """
    Prepares the pre-processed chess sequence corpus from raw data or local cache.

    Checks Hugging Face Hub or local disk cache against the metadata manifest. If missing
    or invalidated, extracts UCI move sequences from source files and writes a new manifest.

    Args:
        cfg: Hydra configuration dictionary.
        root: Optional workspace root directory.

    Returns:
        Tuple of (list_of_uci_sequences, corpus_rebuilt_boolean).
    """
    processed_dir = resolve_path(cfg.data.processed_path, root)
    tokenizer_dir = resolve_path(cfg.data.tokenizer_path, root)
    moves_path = processed_dir / MOVES_NAME
    manifest_path = processed_dir / MANIFEST_NAME
    hf_cfg = _hf_dataset_cfg(cfg)
    force = bool(cfg.data.get("force_reprocess", False))

    if hf_cfg.get("repo_id") and not force:
        pulled = pull_hf_dataset(
            repo_id=str(hf_cfg["repo_id"]),
            revision=str(hf_cfg.get("revision") or "main"),
            processed_dir=processed_dir,
            tokenizer_dir=tokenizer_dir,
        )
        if pulled and _local_cache_hit(cfg, moves_path, manifest_path):
            return _read_moves(moves_path), False

    if _local_cache_hit(cfg, moves_path, manifest_path):
        return _read_moves(moves_path), False

    raw_paths = _ensure_raw_paths(cfg, root)
    sequences = _write_corpus(cfg, raw_paths, moves_path)
    _write_manifest(
        manifest_path,
        {
            **_expected_manifest(cfg, raw_paths),
            "n_sequences": len(sequences),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return sequences, True


def get_tokenizer(
    cfg: DictConfig,
    sequences: list[str] | None = None,
    root: str | Path | None = None,
    corpus_rebuilt: bool = False,
) -> ChessTokenizer:
    """
    Loads an existing ChessTokenizer or trains a new BPE tokenizer on the corpus.

    Args:
        cfg: Hydra configuration dictionary.
        sequences: Optional list of move sequences if tokenizer must be trained from scratch.
        root: Optional workspace root.
        corpus_rebuilt: Boolean flag indicating if corpus was rebuilt, forcing tokenizer retrain.

    Returns:
        Loaded or trained ChessTokenizer instance.
    """
    processed_dir = resolve_path(cfg.data.processed_path, root)
    tokenizer_dir = resolve_path(cfg.data.tokenizer_path, root)
    moves_path = processed_dir / MOVES_NAME
    tokenizer_path = tokenizer_dir / TOKENIZER_NAME
    tokenizer = ChessTokenizer()

    if tokenizer_path.exists() and not corpus_rebuilt:
        tokenizer.load(str(tokenizer_path))
        return tokenizer

    if not moves_path.exists():
        if sequences is None:
            raise ValueError(f"Tokenizer not found at {tokenizer_path} and no sequences provided to train one.")
        moves_path.parent.mkdir(parents=True, exist_ok=True)
        moves_path.write_text("\n".join(sequences) + "\n", encoding="utf-8")

    vocab_size = int(cfg.model.vocab_size)
    tokenizer.train(str(moves_path), vocab_size=vocab_size)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    return tokenizer


def maybe_push_hf_dataset(
    cfg: DictConfig,
    sequences: list[str],
    root: str | Path | None = None,
    corpus_rebuilt: bool = False,
) -> str | None:
    hf_cfg = _hf_dataset_cfg(cfg)
    repo_id = hf_cfg.get("repo_id")
    if not repo_id or not hf_cfg.get("push") or not corpus_rebuilt:
        return None
    return push_hf_dataset(
        repo_id=str(repo_id),
        processed_dir=resolve_path(cfg.data.processed_path, root),
        tokenizer_dir=resolve_path(cfg.data.tokenizer_path, root),
        n_sequences=len(sequences),
        private=True,
    )
