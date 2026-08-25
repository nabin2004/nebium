import io
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path

import chess.pgn
import zstandard as zstd
from omegaconf import DictConfig


def resolve_path(path: str | Path, root: str | Path | None = None) -> Path:
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


def stream_uci_games(cfg: DictConfig, raw_path: Path) -> Iterator[str]:
    data_format = cfg.data.format
    with open_pgn_source(raw_path, data_format) as handle:
        yield from iter_uci_games(handle, int(cfg.data.max_games), int(cfg.data.min_moves))
