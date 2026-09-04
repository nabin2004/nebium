import os
import tempfile
import pytest

from omegaconf import OmegaConf

from src.config.schema import ExperimentConfig, ModelConfig, TrainingConfig, DataConfig, LoggingConfig, HubConfig
from src.data.dataset import build_dataloaders
from src.data.prepare import get_tokenizer, prepare_corpus
from src.models.transformer.nebium import Nebium
from src.training.loop import run_training
from src.utils.reproducibility import seed_everything


def test_end_to_end_integration():
    seed_everything(42)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy PGN data
        pgn_path = os.path.join(tmpdir, "test.pgn")
        with open(pgn_path, "w") as f:
            for _ in range(50):
                f.write("[Event \"Test\"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 *\n\n")
                
        # Build config programmatically based on the dataclasses
        cfg = OmegaConf.create({
            "model": {
                "_target_": "src.models.transformer.nebium.Nebium",
                "vocab_size": 256,
                "d_model": 16,
                "n_heads": 2,
                "n_layers": 1,
                "dropout": 0.0,
                "max_seq_len": 16,
                "bias": True,
            },
            "training": {
                "optimizer": "adamw",
                "learning_rate": 1e-3,
                "weight_decay": 0.1,
                "epochs": 1,
                "batch_size": 4,
                "gradient_accumulation_steps": 1,
                "mixed_precision": "fp32",
                "lr_scheduler": "cosine",
                "warmup_steps": 0,
                "gradient_clipping": 1.0,
                "early_stopping_patience": 1,
            },
            "data": {
                "max_games": 20,
                "raw_path": tmpdir,
                "processed_path": os.path.join(tmpdir, "processed"),
                "tokenizer_path": os.path.join(tmpdir, "tokenizer"),
            },
            "logging": {
                "backend": "null",
                "mode": "disabled",
            },
            "hub": {
                "private": True
            }
        })
        
        # 1. Prepare Corpus
        with open(os.path.join(tmpdir, "mock_urls.txt"), "w") as f:
            f.write(pgn_path + "\n")
        cfg.data.urls = [os.path.join(tmpdir, "mock_urls.txt")]
        # Mocking the download since it's a local file in testing usually, but we skip prepare_corpus if it requires URLs.
        # Instead of prepare_corpus which tries to download, we can directly create the text dataset.
        text_path = os.path.join(tmpdir, "processed", "corpus.txt")
        os.makedirs(os.path.dirname(text_path), exist_ok=True)
        with open(text_path, "w") as f:
            for _ in range(20):
                f.write("e2e4 e7e5 g1f3 b8c6\n")
                
        with open(text_path, "r") as f:
            sequences = f.readlines()
        
        # 2. Tokenizer
        tokenizer = get_tokenizer(cfg, sequences)
        
        # 3. Dataloaders
        train_loader, val_loader = build_dataloaders(
            sequences, tokenizer, cfg.model.max_seq_len, cfg.training.batch_size, 0.8, 42
        )
        
        # 4. Model
        model = Nebium(**cfg.model)
        
        # 5. Train
        from src.logging.factory import build_logger
        logger = build_logger(cfg)
        
        metrics = run_training(model, train_loader, val_loader, cfg, logger)
        
        assert "val/loss" in metrics
        assert "epoch" in metrics
        
        # 6. Generate
        model.eval()
        import torch
        input_ids = torch.tensor([[tokenizer.bos_id]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        out = model.generate(input_ids, attention_mask, max_new_tokens=2, temperature=0.0)
        assert out.shape == (1, 3)
