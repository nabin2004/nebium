import os
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.utils.smoke_test import (
    apply_smoke_test_overrides,
    is_smoke_test,
    preprocess_smoke_test_args,
    verify_smoke_test_artifacts,
)


def test_preprocess_smoke_test_args():
    # Test --smoke-test flag conversion
    args = ["scripts/train.py", "--smoke-test"]
    processed = preprocess_smoke_test_args(args)
    assert "--smoke-test" not in processed
    assert "++smoke_test=true" in processed

    # Test alternate forms
    assert "++smoke_test=true" in preprocess_smoke_test_args(["--smoke_test"])
    assert "++smoke_test=true" in preprocess_smoke_test_args(["-s"])

    # Test explicit false
    assert "++smoke_test=true" not in preprocess_smoke_test_args(["--smoke-test=false"])

    # Test preservation of other args
    mixed = preprocess_smoke_test_args(["--config-name", "kaggle", "--smoke-test", "training.batch_size=8"])
    assert "--smoke-test" not in mixed
    assert "--config-name" in mixed
    assert "kaggle" in mixed
    assert "training.batch_size=8" in mixed
    assert "++smoke_test=true" in mixed


def test_apply_smoke_test_overrides():
    cfg = OmegaConf.create({
        "data": {
            "urls": ["https://example.com/dump.pgn.zst"],
            "raw_path": "data/raw",
            "format": "pgn.zst",
            "max_games": 200000,
            "min_moves": 20,
            "train_split": 0.9,
            "processed_path": "data/processed/full",
            "tokenizer_path": "data/tokenizer/full",
            "hf_dataset": {"push": True, "repo_id": "test/repo"},
        },
        "training": {
            "epochs": 20,
            "batch_size": 64,
            "warmup_steps": 1000,
            "gradient_accumulation_steps": 4,
            "early_stopping_patience": 5,
            "export_gguf": True,
            "generate_report": True,
        },
        "model": {
            "n_layers": 6,
            "d_model": 512,
        },
        "logging": {
            "backend": "wandb",
            "mode": "online",
        },
        "hub": {
            "push": True,
            "repo_id": "test/model",
        }
    })

    apply_smoke_test_overrides(cfg)

    assert is_smoke_test(cfg) is True
    assert cfg.training.epochs == 1
    assert cfg.training.batch_size <= 4
    assert cfg.training.warmup_steps == 1
    assert cfg.training.gradient_accumulation_steps == 1
    assert cfg.training.generate_report is False
    assert cfg.data.format == "pgn"
    assert cfg.data.max_games <= 20
    assert cfg.data.min_moves <= 5
    assert cfg.data.force_reprocess is True
    assert cfg.data.hf_dataset.push is False
    assert cfg.hub.push is False


def test_verify_smoke_test_artifacts(tmp_path):
    # Missing artifacts should report False
    results = verify_smoke_test_artifacts(tmp_path)
    assert not all(results.values())

    # Create dummy artifacts
    (tmp_path / "checkpoint.pt").write_text("dummy")
    (tmp_path / "best_model.pt").write_text("dummy")
    (tmp_path / "training_history.json").write_text("[]")
    (tmp_path / "nebium.gguf").write_text("dummy")

    export_dir = tmp_path / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "model.pt").write_text("dummy")
    (export_dir / "model_config.json").write_text("{}")
    (export_dir / "tokenizer.json").write_text("{}")
    (export_dir / "README.md").write_text("# Model Card")

    results_complete = verify_smoke_test_artifacts(tmp_path)
    assert all(results_complete.values())


@pytest.mark.integration
def test_smoke_test_cli_default_config():
    cmd = [sys.executable, "scripts/train.py", "--smoke-test"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"train.py --smoke-test failed: {result.stderr}"
    assert "[PASS] [NEBIUM SMOKE TEST] ALL PIPELINE STAGES PASSED SUCCESSFULLY!" in result.stdout


@pytest.mark.integration
def test_smoke_test_cli_kaggle_config():
    cmd = [sys.executable, "scripts/train.py", "--config-name", "kaggle", "--smoke-test"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"train.py --config-name kaggle --smoke-test failed: {result.stderr}"
    assert "[PASS] [NEBIUM SMOKE TEST] ALL PIPELINE STAGES PASSED SUCCESSFULLY!" in result.stdout


@pytest.mark.integration
def test_smoke_test_cli_prepare_data():
    cmd = [sys.executable, "scripts/prepare_data.py", "--smoke-test"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"prepare_data.py --smoke-test failed: {result.stderr}"
    assert "[PASS] [Data Smoke Test]" in result.stdout
