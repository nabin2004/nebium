"""
Data source resolution and streaming utilities for Nebium.

Handles path resolution, decompression of .pgn and .pgn.zst archives, and
streaming parsing of PGN game headers and move nodes.
"""

import io
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import chess.pgn
import zstandard as zstd
from omegaconf import DictConfig


def resolve_path(path: str | Path, root: str | Path | None = None) -> Path:
    """
    Resolves relative file paths against the optional project root directory.

    Args:
        path: Relative or absolute Path or path string.
        root: Optional workspace root directory.

    Returns:
        Fully resolved Path object.
    """
    resolved = Path(path)
    if not resolved.is_absolute() and root is not None:
        resolved = Path(root) / resolved
    return resolved



def resolve_raw_path(raw_path: Path, data_format: str) -> Path:
    if raw_path.is_file():
        return raw_path
    if raw_path.is_dir():
        pattern = "*.pgn.zst" if data_format == "pgn_zst" else "*.pgn"
        matches = sorted(raw_path.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No {pattern} files under {raw_path}")
        return matches[0]
    raise FileNotFoundError(f"Raw data path does not exist: {raw_path}")


@contextmanager
def open_pgn_source(path: Path, data_format: str):
    if data_format == "pgn":
        with path.open(encoding="utf-8") as handle:
            yield handle
        return
    if data_format == "pgn_zst":
        raw = path.open("rb")
        try:
            with zstd.ZstdDecompressor().stream_reader(raw) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8") as handle:
                    yield handle
        finally:
            raw.close()
        return
    raise ValueError(f"Unsupported data format: {data_format}")


def iter_uci_games(handle, max_games: int, min_moves: int) -> Iterator[str]:
    kept = 0
    while kept < max_games:
        game = chess.pgn.read_game(handle)
        if game is None:
            break
        board = game.board()
        moves: list[str] = []
        for move in game.mainline_moves():
            moves.append(board.uci(move))
            board.push(move)
        if len(moves) < min_moves:
            continue
        kept += 1
        yield " ".join(moves)


def resolve_raw_files(raw_path: Path, data_format: str, preferred: list[Path] | None = None) -> list[Path]:
    if preferred:
        existing = [path for path in preferred if path.exists() and path.stat().st_size > 0]
        if existing:
            return existing
    if raw_path.is_file():
        return [raw_path]
    if raw_path.is_dir():
        pattern = "*.pgn.zst" if data_format == "pgn_zst" else "*.pgn"
        matches = sorted(raw_path.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No {pattern} files under {raw_path}")
        return matches
    raise FileNotFoundError(f"Raw data path does not exist: {raw_path}")


def stream_uci_games(cfg: DictConfig, raw_paths: Path | list[Path]) -> Iterator[str]:
    paths = [raw_paths] if isinstance(raw_paths, Path) else list(raw_paths)
    remaining = int(cfg.data.max_games)
    min_moves = int(cfg.data.min_moves)
    data_format = cfg.data.format
    for path in paths:
        if remaining <= 0:
            break
        with open_pgn_source(path, data_format) as handle:
            for sequence in iter_uci_games(handle, remaining, min_moves):
                remaining -= 1
                yield sequence
