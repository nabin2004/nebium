import argparse
import os
import chess
import torch
import hydra
from hydra.utils import instantiate
from hydra import compose, initialize
from src.data.prepare import get_tokenizer

# Each entry: (prompt_moves_str, expected_next_move_uci)
# The model is given the prompt and must predict the expected next move.
OPENINGS = {
    "Ruy Lopez (3...a6)": ("e2e4 e7e5 g1f3 b8c6 f1b5", "a7a6"),
    "Sicilian Defense (2.Nf3)": ("e2e4 c7c5", "g1f3"),
    "Queen's Gambit (2...dxc4 or 2...e6)": ("d2d4 d7d5 c2c4", "d5c4"),  # most common
    "French Defense (2.d4)": ("e2e4 e7e6", "d2d4"),
    "Caro-Kann (2.d4)": ("e2e4 c7c6", "d2d4"),
    "Italian Game (3...Bc5)": ("e2e4 e7e5 g1f3 b8c6 f1c4", "f8c5"),
}

def get_nebium_move(model, tokenizer, board, device):
    prompt_str = " ".join([m.uci() for m in board.move_stack])
    if prompt_str:
        encoded = tokenizer.encode(prompt_str)
        input_ids = [tokenizer.bos_id] + encoded
    else:
        input_ids = [tokenizer.bos_id]
        
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(x, dtype=torch.long, device=device)
    
    with torch.no_grad():
        logits = model(x, mask)
        next_token_logits = logits[0, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)
        sorted_indices = torch.argsort(probs, descending=True)
        
        for idx in sorted_indices:
            token_id = idx.item()
            if token_id in (tokenizer.eos_id, tokenizer.pad_id):
                continue
            token_str = tokenizer.decode([token_id]).strip()
            if not token_str:
                continue
            m_str = token_str.split()[0]
            try:
                move = chess.Move.from_uci(m_str)
                if move in board.legal_moves:
                    return move
            except Exception:
                continue
    import random
    return random.choice(list(board.legal_moves))

def evaluate_opening(model, tokenizer, device, opening_name, prompt_moves_str, expected_move):
    board = chess.Board()
    for m in prompt_moves_str.split():
        board.push(chess.Move.from_uci(m))
        
    predicted = get_nebium_move(model, tokenizer, board, device)
    
    is_correct = (predicted.uci() == expected_move)
    print(f"[{opening_name}] Prompt: {prompt_moves_str!r}")
    print(f"  Expected: {expected_move}, Predicted: {predicted.uci()} -> {'PASS' if is_correct else 'FAIL'}")
    return is_correct

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="nebium_stub")
    parser.add_argument("--checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--output", type=str, default="", help="Path to save JSON results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        initialize(version_base=None, config_path="../configs")
    cfg = compose(config_name="config", overrides=[f"model={args.config_name}"])
    
    tokenizer = get_tokenizer(cfg)
    cfg.model.vocab_size = tokenizer.vocab_size
    model = instantiate(cfg.model)
    
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt.get("model_state_dict")))
        
    model.to(device)
    model.eval()

    correct = 0
    results_dict = {}
    for name, (prompt_moves, expected_move) in OPENINGS.items():
        is_correct = evaluate_opening(model, tokenizer, device, name, prompt_moves, expected_move)
        if is_correct:
            correct += 1
        results_dict[name] = {"expected": expected_move, "correct": is_correct}
            
    score = correct / len(OPENINGS)
    print(f"\nOpening Compliance Score: {correct}/{len(OPENINGS)} ({score*100:.1f}%)")
    
    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump({"correct": correct, "total": len(OPENINGS), "score": score, "details": results_dict}, f, indent=4)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
