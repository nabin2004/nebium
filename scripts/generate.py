import argparse
import torch
import hydra
from omegaconf import DictConfig

from src.data.prepare import get_tokenizer
from src.utils.reproducibility import seed_everything


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    parser = argparse.ArgumentParser(description="Generate chess moves using a trained Nebium model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--prompt", type=str, default="", help="Initial moves (e.g. 'e2e4 e7e5')")
    parser.add_argument("--max_moves", type=int, default=10, help="Maximum number of moves to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    
    # We use unknown args for argparse to avoid hydra overrides crashing it
    import sys
    args, _ = parser.parse_known_args(sys.argv[1:])
    
    seed_everything(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint {args.checkpoint} on {device}...")
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    tokenizer = get_tokenizer(cfg)
    
    # Initialize model using config from checkpoint or current config
    # We will instantiate current config model, then load state dict
    from hydra.utils import instantiate
    model = instantiate(cfg.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    prompt = args.prompt.strip()
    
    # Encode prompt, prepend BOS
    if prompt:
        input_ids = [tokenizer.bos_id] + tokenizer.encode(prompt)
    else:
        input_ids = [tokenizer.bos_id]
        
    print(f"\nInitial prompt: '{prompt}'")
    
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor, device=device)
    
    with torch.no_grad():
        output = model.generate(
            input_tensor,
            attention_mask,
            max_new_tokens=args.max_moves,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
    
    output_ids = output[0].tolist()
    decoded = tokenizer.decode(output_ids)
    print(f"\nGenerated sequence:\n{decoded}")


if __name__ == "__main__":
    main()
