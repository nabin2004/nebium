import json
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import DictConfig

from src.data.sources import resolve_path, resolve_raw_path, stream_uci_games
from src.data.tokenizer import ChessTokenizer

MOVES_NAME = "moves.txt"
MANIFEST_NAME = "manifest.json"
TOKENIZER_NAME = "tokenizer.json"


def _source_identity(path: Path) -> dict:
    stat = path.stat()
    return {
        "source": str(path.resolve()),
        "source_size": stat.st_size,
        "source_mtime": stat.st_mtime,
    }


def _expected_manifest(cfg: DictConfig, raw_path: Path) -> dict:
    return {
        **_source_identity(raw_path),
        "format": cfg.data.format,
        "max_games": int(cfg.data.max_games),
        "min_moves": int(cfg.data.min_moves),
    }


def _manifest_matches(existing: dict, expected: dict) -> bool:
    for key in ("source", "source_size", "format", "max_games", "min_moves"):
        if existing.get(key) != expected.get(key):
            return False
    return abs(float(existing.get("source_mtime", 0)) - float(expected["source_mtime"])) < 1e-3


def _read_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_moves(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_corpus(cfg: DictConfig, raw_path: Path, moves_path: Path) -> list[str]:
    moves_path.parent.mkdir(parents=True, exist_ok=True)
    sequences: list[str] = []
    with moves_path.open("w", encoding="utf-8") as out:
        for sequence in stream_uci_games(cfg, raw_path):
            out.write(sequence + "\n")
            sequences.append(sequence)
    return sequences


def prepare_corpus(cfg: DictConfig, root: str | Path | None = None) -> tuple[list[str], bool]:
    raw_path = resolve_raw_path(resolve_path(cfg.data.raw_path, root), cfg.data.format)
    processed_dir = resolve_path(cfg.data.processed_path, root)
    moves_path = processed_dir / MOVES_NAME
    manifest_path = processed_dir / MANIFEST_NAME
    expected = _expected_manifest(cfg, raw_path)
    force = bool(cfg.data.get("force_reprocess", False))

    existing = _read_manifest(manifest_path)
    cache_hit = (
        not force
        and moves_path.exists()
        and existing is not None
        and _manifest_matches(existing, expected)
    )
    if cache_hit:
        sequences = _read_moves(moves_path)
        return sequences, False

    sequences = _write_corpus(cfg, raw_path, moves_path)
    _write_manifest(
        manifest_path,
        {
            **expected,
            "n_sequences": len(sequences),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return sequences, True


def get_tokenizer(
    cfg: DictConfig,
    sequences: list[str],
    root: str | Path | None = None,
    corpus_rebuilt: bool = False,
) -> ChessTokenizer:
    processed_dir = resolve_path(cfg.data.processed_path, root)
    tokenizer_dir = resolve_path(cfg.data.tokenizer_path, root)
    moves_path = processed_dir / MOVES_NAME
    tokenizer_path = tokenizer_dir / TOKENIZER_NAME
    tokenizer = ChessTokenizer()

    if tokenizer_path.exists() and not corpus_rebuilt:
        tokenizer.load(str(tokenizer_path))
        return tokenizer

    if not moves_path.exists():
        moves_path.parent.mkdir(parents=True, exist_ok=True)
        moves_path.write_text("\n".join(sequences) + "\n", encoding="utf-8")

    vocab_size = int(cfg.model.vocab_size)
    tokenizer.train(str(moves_path), vocab_size=vocab_size)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    return tokenizer
