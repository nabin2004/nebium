import os
import sys
import time
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf

from src.config.schema import register_configs
from src.data.dataset import build_dataloaders
from src.data.prepare import get_tokenizer, maybe_push_hf_dataset, prepare_corpus
from src.export.gguf_export import export_to_gguf
from src.hub.push import export_checkpoint, push_to_hub
from src.logging.factory import build_logger
from src.training.loop import run_training
from src.utils.kaggle import setup_kaggle_env
from src.utils.reproducibility import seed_everything
from src.utils.smoke_test import (
    apply_smoke_test_overrides,
    is_smoke_test,
    preprocess_smoke_test_args,
    print_smoke_test_banner,
    print_smoke_test_summary,
    verify_smoke_test_artifacts,
)

# Intercept and translate CLI flags like --smoke-test before Hydra parses sys.argv
sys.argv[1:] = preprocess_smoke_test_args(sys.argv[1:])

register_configs()


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    smoke = is_smoke_test(cfg)
    smoke_start_time = time.time()

    if smoke:
        print_smoke_test_banner()
        root = get_original_cwd()
        apply_smoke_test_overrides(cfg, root=root)

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

        hub_status = None
        if smoke:
            # Fully exercise checkpoint packaging and export directory generation
            try:
                export_checkpoint(model, tokenizer, cfg, metrics, Path("export"))
            except Exception as exc:
                print(f"[Smoke Test] Checkpoint export warning: {exc}")

            # Check and report HF credentials if available
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            repo_id = cfg.hub.get("repo_id")
            if token:
                try:
                    from huggingface_hub import HfApi
                    user_info = HfApi().whoami(token=token)
                    user_name = user_info.get("name", "user")
                    target_repo = repo_id if repo_id else f"{user_name}/nebium"
                    hub_status = f"Auth valid (@{user_name}, target: {target_repo}). Push skipped during smoke test."
                except Exception as exc:
                    hub_status = f"HF_TOKEN detected, but auth check failed: {exc}"
            elif repo_id:
                hub_status = f"Target repo set ({repo_id}), but no HF_TOKEN found in environment."
            else:
                hub_status = "No HF_TOKEN configured. Set Kaggle secret 'HF_TOKEN' for production pushes."
        else:
            repo_id = push_to_hub(model, tokenizer, cfg, metrics)
            if repo_id:
                print(f"Pushed checkpoint to https://huggingface.co/{repo_id}")

        if not smoke and bool(cfg.training.get("generate_report", True)):
            import subprocess
            print("\nGenerating final paper report and running extensive evaluations...")
            subprocess.run(f"{sys.executable} scripts/generate_paper_report.py", shell=True)

        if smoke:
            duration = time.time() - smoke_start_time
            artifacts = verify_smoke_test_artifacts(".")
            success = print_smoke_test_summary(metrics, artifacts, duration, hub_status=hub_status)
            if not success:
                raise RuntimeError("Smoke test artifact or metrics verification failed!")

    finally:
        logger.finish()


if __name__ == "__main__":
    main()
