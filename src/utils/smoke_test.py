import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.utils.kaggle import is_kaggle_environment


SMOKE_TEST_CLI_FLAGS = {"--smoke-test", "--smoke_test", "-s"}


def preprocess_smoke_test_args(args: list[str] | None = None) -> list[str]:
    """
    Preprocesses CLI arguments to translate '--smoke-test' flags into Hydra overrides.
    Also handles environment compatibility (e.g. running kaggle.yaml outside Kaggle).
    """
    if args is None:
        args = sys.argv[1:]
    else:
        args = list(args)

    smoke_test_requested = False
    cleaned_args: list[str] = []

    for arg in args:
        if arg in SMOKE_TEST_CLI_FLAGS or arg.startswith("--smoke-test=") or arg.startswith("--smoke_test="):
            smoke_test_requested = True
            # Handle explicit boolean value if provided
            if "=" in arg:
                val = arg.split("=", 1)[1].lower()
                if val in ("false", "0", "no"):
                    smoke_test_requested = False
        else:
            cleaned_args.append(arg)

    if smoke_test_requested:
        # Check if already specified in overrides
        has_override = any("smoke_test" in a for a in cleaned_args)
        if not has_override:
            cleaned_args.append("++smoke_test=true")

    # If running with kaggle config outside Kaggle, redirect hydra output dir to local outputs
    # unless already explicitly overridden
    is_kaggle_config = any("kaggle" in a for a in cleaned_args)
    has_hydra_dir = any(a.startswith("hydra.run.dir=") or a.startswith("+hydra.run.dir=") for a in cleaned_args)
    if is_kaggle_config and not is_kaggle_environment() and not has_hydra_dir:
        cleaned_args.append("hydra.run.dir=outputs/${now:%Y-%m-%d_%H-%M-%S}")

    return cleaned_args


def is_smoke_test(cfg: DictConfig) -> bool:
    """Checks whether the configuration has smoke_test enabled."""
    return bool(cfg.get("smoke_test", False))


def apply_smoke_test_overrides(cfg: DictConfig, root: Path | str | None = None) -> None:
    """
    Applies aggressive, safe overrides to the Hydra configuration for a fast (<10s)
    end-to-end verification of all pipeline stages.
    """
    OmegaConf.set_struct(cfg, False)
    cfg.smoke_test = True

    root_path = Path(root) if root else Path.cwd()

    # 1. Safe Data configuration
    # Locate sample fixture PGN
    sample_pgn = root_path / "data" / "fixtures" / "sample.pgn"
    if not sample_pgn.exists():
        # Fallback search if called from a subdirectory or parent
        sample_pgn = Path("data/fixtures/sample.pgn")

    cfg.data.urls = []
    if sample_pgn.exists():
        cfg.data.raw_path = str(sample_pgn)
    cfg.data.format = "pgn"
    cfg.data.max_games = min(int(cfg.data.get("max_games", 20)), 20)
    cfg.data.min_moves = min(int(cfg.data.get("min_moves", 5)), 5)
    cfg.data.force_reprocess = True
    cfg.data.train_split = 0.8

    # Separate cache directories so smoke tests never clobber production processed data
    if is_kaggle_environment():
        cfg.data.processed_path = "/kaggle/working/data/processed/smoke_test"
        cfg.data.tokenizer_path = "/kaggle/working/data/tokenizer/smoke_test"
    else:
        cfg.data.processed_path = "data/processed/smoke_test"
        cfg.data.tokenizer_path = "data/tokenizer/smoke_test"

    # Never push dummy datasets during smoke test
    if "hf_dataset" in cfg.data:
        cfg.data.hf_dataset.push = False

    # 2. Training configuration (fast 1 epoch, small batch, limited steps)
    cfg.training.epochs = 1
    cfg.training.batch_size = min(int(cfg.training.get("batch_size", 4)), 4)
    cfg.training.warmup_steps = 1
    cfg.training.gradient_accumulation_steps = 1
    cfg.training.early_stopping_patience = 1

    # Evaluation knobs
    cfg.training.eval_samples = True
    raw_prompts = cfg.training.get("sample_prompts", ["", "e2e4"])
    cfg.training.sample_prompts = list(raw_prompts)[:2] if raw_prompts else ["", "e2e4"]
    cfg.training.sample_max_moves = min(int(cfg.training.get("sample_max_moves", 5)), 5)
    cfg.training.eval_puzzles = True
    puzzle_fixture = root_path / "data" / "fixtures" / "puzzles.jsonl"
    if puzzle_fixture.exists():
        cfg.training.puzzle_path = str(puzzle_fixture)

    # Local GGUF export
    cfg.training.export_gguf = True
    # Disable long stockfish paper report during quick smoke test
    cfg.training.generate_report = False

    # 3. Model configuration (ensure rapid execution on CPU if CUDA is unavailable)
    import torch

    if not torch.cuda.is_available() and int(cfg.model.get("n_layers", 1)) > 2:
        cfg.model.n_layers = 2

    # 4. Logging configuration
    # Prevent crashing on missing WandB auth or polluting cloud runs
    if cfg.logging.get("backend") == "wandb":
        has_wandb_key = bool(os.environ.get("WANDB_API_KEY"))
        if not has_wandb_key:
            cfg.logging.mode = "disabled"
        elif cfg.logging.get("mode") == "online":
            cfg.logging.mode = "offline"

    # 5. Hub configuration
    # Validate credentials & packaging, but skip live push to protect production repo
    if "hub" in cfg:
        cfg.hub.push = False
        cfg.hub.export_gguf = True


def verify_smoke_test_artifacts(dest_dir: Path | str = ".") -> dict[str, bool]:
    """
    Verifies that all pipeline artifacts were generated properly and are non-empty.
    """
    dest = Path(dest_dir)
    export_dir = dest / "export"

    artifacts = {
        "checkpoint.pt": (dest / "checkpoint.pt").exists() and (dest / "checkpoint.pt").stat().st_size > 0,
        "best_model.pt": (dest / "best_model.pt").exists() and (dest / "best_model.pt").stat().st_size > 0,
        "training_history.json": (dest / "training_history.json").exists() and (dest / "training_history.json").stat().st_size > 0,
        "nebium.gguf": (dest / "nebium.gguf").exists() and (dest / "nebium.gguf").stat().st_size > 0,
        "export/model.pt": (export_dir / "model.pt").exists() and (export_dir / "model.pt").stat().st_size > 0,
        "export/model_config.json": (export_dir / "model_config.json").exists() and (export_dir / "model_config.json").stat().st_size > 0,
        "export/tokenizer.json": (export_dir / "tokenizer.json").exists() and (export_dir / "tokenizer.json").stat().st_size > 0,
        "export/README.md": (export_dir / "README.md").exists() and (export_dir / "README.md").stat().st_size > 0,
    }
    return artifacts


def print_smoke_test_banner() -> None:
    print("\n" + "=" * 70)
    print("  [NEBIUM SMOKE TEST] Running End-to-End Pipeline Verification")
    print("=" * 70)
    print("  Mode       : Fast verification (<10s)")
    print("  Scope      : Data Ingestion -> Tokenizer -> Model -> Train -> Eval")
    print("               -> Checkpoint -> GGUF Export -> Hub Packaging")
    print("=" * 70 + "\n")


def print_smoke_test_summary(
    metrics: dict[str, Any],
    artifacts: dict[str, bool],
    duration: float,
    hub_status: str | None = None,
) -> bool:
    all_artifacts_ok = all(artifacts.values())
    has_metrics = bool(metrics and "val/loss" in metrics)
    success = all_artifacts_ok and has_metrics

    print("\n" + "=" * 70)
    if success:
        print("  [PASS] [NEBIUM SMOKE TEST] ALL PIPELINE STAGES PASSED SUCCESSFULLY!")
    else:
        print("  [FAIL] [NEBIUM SMOKE TEST] SOME PIPELINE CHECKS FAILED!")
    print("=" * 70)
    print(f"  Total Duration       : {duration:.2f}s")
    print(f"  Validation Loss      : {metrics.get('val/loss', 'N/A')}")
    print(f"  Validation Accuracy  : {metrics.get('val/accuracy', 'N/A')}")
    print(f"  Legal Move Rate      : {metrics.get('val/legal_move_rate', 'N/A')}")
    print(f"  Puzzle Accuracy      : {metrics.get('val/puzzle_accuracy', 'N/A')}")
    if hub_status:
        print(f"  Hugging Face Hub     : {hub_status}")
    print("-" * 70)
    print("  Artifact Verification:")
    for name, ok in artifacts.items():
        status_str = "[OK] Present" if ok else "[FAIL] Missing/Empty"
        print(f"    - {name:<26} : {status_str}")
    print("=" * 70 + "\n")

    return success
