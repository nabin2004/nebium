import argparse
from pathlib import Path

import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from src.data.prepare import get_tokenizer
from src.export.gguf_export import export_to_gguf


def main():
    parser = argparse.ArgumentParser(description="Export trained Nebium checkpoint to GGUF format")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--output", type=str, default="nebium.gguf", help="Output path for .gguf file")
    parser.add_argument("--precision", type=str, choices=["fp16", "fp32"], default="fp16", help="Precision (fp16 or fp32)")
    parser.add_argument("--config-name", type=str, default="config", help="Hydra config name (default: config)")
    args, unknown = parser.parse_known_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name=args.config_name, overrides=unknown)

    # Load tokenizer
    tokenizer = get_tokenizer(cfg)
    cfg.model.vocab_size = tokenizer.vocab_size

    # Instantiate model and load state_dict
    model = instantiate(cfg.model)
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(state_dict)
    model.eval()

    # Export to GGUF
    export_to_gguf(
        model=model,
        tokenizer=tokenizer,
        output_path=args.output,
        precision=args.precision,
    )


if __name__ == "__main__":
    main()
