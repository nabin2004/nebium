import sys

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from src.config.schema import register_configs
from src.data.prepare import get_tokenizer, maybe_push_hf_dataset, prepare_corpus
from src.utils.smoke_test import (
    apply_smoke_test_overrides,
    is_smoke_test,
    preprocess_smoke_test_args,
    print_smoke_test_banner,
)

# Intercept and translate CLI flags like --smoke-test before Hydra parses sys.argv
sys.argv[1:] = preprocess_smoke_test_args(sys.argv[1:])

register_configs()


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    smoke = is_smoke_test(cfg)
    root = get_original_cwd()
    if smoke:
        print_smoke_test_banner()
        apply_smoke_test_overrides(cfg, root=root)

    sequences, rebuilt = prepare_corpus(cfg, root=root)
    if not sequences:
        raise RuntimeError("No games loaded. Check data.raw_path, format, min_moves, and max_games.")
    tokenizer = get_tokenizer(cfg, sequences, root=root, corpus_rebuilt=rebuilt)
    dataset_repo = maybe_push_hf_dataset(cfg, sequences, root=root, corpus_rebuilt=rebuilt)
    action = "rebuilt" if rebuilt else "reused"
    print(
        f"Corpus {action}: {len(sequences)} games, vocab_size={tokenizer.vocab_size} "
        f"({cfg.data.processed_path}, {cfg.data.tokenizer_path})"
    )
    if dataset_repo:
        print(f"Pushed processed dataset to https://huggingface.co/datasets/{dataset_repo}")
    if smoke:
        print("[PASS] [Data Smoke Test] Successfully prepared corpus and trained tokenizer!")


if __name__ == "__main__":
    main()
