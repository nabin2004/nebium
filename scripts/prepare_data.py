import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from src.data.prepare import get_tokenizer, prepare_corpus


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    root = get_original_cwd()
    sequences, rebuilt = prepare_corpus(cfg, root=root)
    if not sequences:
        raise RuntimeError("No games loaded. Check data.raw_path, format, min_moves, and max_games.")
    tokenizer = get_tokenizer(cfg, sequences, root=root, corpus_rebuilt=rebuilt)
    action = "rebuilt" if rebuilt else "reused"
    print(
        f"Corpus {action}: {len(sequences)} games, vocab_size={tokenizer.vocab_size} "
        f"({cfg.data.processed_path}, {cfg.data.tokenizer_path})"
    )


if __name__ == "__main__":
    main()
