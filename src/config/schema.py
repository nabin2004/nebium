"""
Hydra configuration schemas and structured dataclasses for Nebium.

Defines strongly typed configurations for model architecture, data pipeline,
optimizer/scheduler training parameters, evaluation, and experiment tracking.
"""

from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore


@dataclass
class ModelConfig:
    """
    Hyperparameters defining the Nebium neural network architecture.
    """
    _target_: str = "src.models.transformer.nebium.Nebium"
    vocab_size: int = 5000
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 1
    dropout: float = 0.1
    positional_encoding: str = "rope"
    attention_type: str = "standard"
    activation: str = "swiglu"
    norm: str = "rmsnorm"
    max_seq_len: int = 128
    bias: bool = False
    tie_word_embeddings: bool = False



@dataclass
class HFDatasetConfig:
    repo_id: Optional[str] = None
    push: bool = False
    revision: str = "main"


@dataclass
class DataConfig:
    urls: list[str] = field(default_factory=list)
    format: str = "pgn.zst"
    max_games: int = 1000
    min_moves: int = 10
    raw_path: str = "data/raw"
    processed_path: str = "data/processed/default"
    tokenizer_path: str = "data/tokenizer/default"
    train_split: float = 0.9
    seed: int = 42
    force_reprocess: bool = False
    hf_dataset: HFDatasetConfig = field(default_factory=HFDatasetConfig)
    representation: str = "uci"


@dataclass
class TrainingConfig:
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    epochs: int = 1
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "fp16"
    lr_scheduler: str = "cosine"
    warmup_steps: int = 100
    gradient_clipping: float = 1.0
    early_stopping_patience: int = 5
    resume_from: Optional[str] = None
    eval_samples: bool = True
    sample_prompts: list[str] = field(default_factory=lambda: ["", "e2e4", "d2d4"])
    sample_max_moves: int = 20
    sample_temperature: float = 0.7
    export_gguf: bool = True
    gguf_precision: str = "fp16"


@dataclass
class LoggingConfig:
    backend: str = "wandb"
    mode: str = "online"
    project: str = "nebium"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class HubConfig:
    repo_id: Optional[str] = None
    private: bool = True
    push: bool = False
    push_on_epoch: bool = False
    commit_message: str = "Add Nebium checkpoint"
    export_gguf: bool = True
    gguf_precision: str = "fp16"


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    hub: HubConfig = field(default_factory=HubConfig)
    smoke_test: bool = False


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=ExperimentConfig)
    cs.store(group="model", name="base_model", node=ModelConfig)
    cs.store(group="training", name="base_training", node=TrainingConfig)
    cs.store(group="data", name="base_data", node=DataConfig)
    cs.store(group="logging", name="base_logging", node=LoggingConfig)
    cs.store(group="hub", name="base_hub", node=HubConfig)
