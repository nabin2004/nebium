import hydra
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf

from src.config.schema import register_configs
from src.data.dataset import build_dataloaders
from src.data.prepare import get_tokenizer, maybe_push_hf_dataset, prepare_corpus
from src.export.gguf_export import export_to_gguf
from src.hub.push import push_to_hub
from src.logging.factory import build_logger
from src.training.loop import run_training
from src.utils.kaggle import setup_kaggle_env
from src.utils.reproducibility import seed_everything

register_configs()


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Auto-configure Kaggle secrets and directories before logger initialization
    setup_kaggle_env()

    seed_everything(int(cfg.data.seed))
    logger = build_logger(cfg)
    try:
        logger.log_config(cfg)
        root = get_original_cwd()
        sequences, rebuilt = prepare_corpus(cfg, root=root)
        if not sequences:
            raise RuntimeError("No games loaded. Check data.raw_path, format, min_moves, and max_games.")
        tokenizer = get_tokenizer(cfg, sequences, root=root, corpus_rebuilt=rebuilt)
        dataset_repo = maybe_push_hf_dataset(cfg, sequences, root=root, corpus_rebuilt=rebuilt)
        if dataset_repo:
            print(f"Pushed processed dataset to https://huggingface.co/datasets/{dataset_repo}")
        OmegaConf.set_struct(cfg, False)
        cfg.model.vocab_size = tokenizer.vocab_size
        model = instantiate(cfg.model)
        train_loader, val_loader = build_dataloaders(
            sequences,
            tokenizer,
            max_seq_len=int(cfg.model.max_seq_len),
            batch_size=int(cfg.training.batch_size),
            train_split=float(cfg.data.train_split),
            seed=int(cfg.data.seed),
        )
        metrics = run_training(model, train_loader, val_loader, cfg, logger, tokenizer=tokenizer)

        # Local GGUF export if enabled
        if bool(cfg.training.get("export_gguf", True)):
            try:
                export_to_gguf(
                    model=model,
                    tokenizer=tokenizer,
                    output_path="nebium.gguf",
                    precision=str(cfg.training.get("gguf_precision", "fp16")),
                )
            except Exception as exc:
                print(f"[GGUF Export] Warning: Failed to export local GGUF: {exc}")

        repo_id = push_to_hub(model, tokenizer, cfg, metrics)
        if repo_id:
            print(f"Pushed checkpoint to https://huggingface.co/{repo_id}")

        if bool(cfg.training.get("generate_report", True)):
            import subprocess
            print("\nGenerating final paper report and running extensive evaluations...")
            subprocess.run("uv run python scripts/generate_paper_report.py", shell=True)

    finally:
        logger.finish()

if __name__ == "__main__":
    main()
