import argparse
import os
import urllib.request
import zipfile
import chess
import chess.engine
import torch
import hydra
from hydra.utils import instantiate
from hydra import compose, initialize
from src.data.prepare import get_tokenizer

def ensure_stockfish():
    sf_path = "stockfish_dir/stockfish/stockfish-windows-x86-64-avx2.exe"
    if os.path.exists(sf_path):
        return sf_path
        
    print("Downloading Stockfish...")
    url = "https://github.com/official-stockfish/Stockfish/releases/download/sf_16.1/stockfish-windows-x86-64-avx2.zip"
    urllib.request.urlretrieve(url, "stockfish.zip")
    with zipfile.ZipFile("stockfish.zip", 'r') as zip_ref:
        zip_ref.extractall("stockfish_dir")
    print("Stockfish downloaded.")
    return sf_path

def get_nebium_move(model, tokenizer, board, device):
    # Predict the next move given the board
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
        
        # We need a legal move. Let's do greedy decoding but restricted to legal moves
        probs = torch.softmax(next_token_logits, dim=-1)
        sorted_indices = torch.argsort(probs, descending=True)
        
        for idx in sorted_indices:
            token_id = idx.item()
            if token_id in (tokenizer.eos_id, tokenizer.pad_id):
                continue
            
            token_str = tokenizer.decode([token_id]).strip()
            if not token_str:
                continue
                
            # Token might contain multiple moves. We take the first one.
            m_str = token_str.split()[0]
            try:
                move = chess.Move.from_uci(m_str)
                if move in board.legal_moves:
                    return move
            except Exception:
                continue
                
    # Fallback to random legal move if model fails to produce one
    import random
    return random.choice(list(board.legal_moves))

def main():
    parser = argparse.ArgumentParser(description="Self-play against Stockfish for Elo Estimation")
    parser.add_argument("--config_name", type=str, default="nebium_stub", help="Model configuration")
    parser.add_argument("--checkpoint", type=str, default="best_model.pt", help="Path to checkpoint")
    parser.add_argument("--games", type=int, default=10, help="Number of games to play")
    parser.add_argument("--depth", type=int, default=10, help="Stockfish depth")
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
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint.get("model", checkpoint.get("model_state_dict")))
    else:
        print("Warning: Checkpoint not found. Using randomly initialized weights.")
        
    model.to(device)
    model.eval()

    sf_path = ensure_stockfish()
    
    wins = 0
    losses = 0
    draws = 0
    
    print(f"Starting {args.games} games against Stockfish at depth {args.depth}...")
    
    with chess.engine.SimpleEngine.popen_uci(sf_path) as engine:
        for i in range(args.games):
            board = chess.Board()
            nebium_is_white = (i % 2 == 0)
            
            while not board.is_game_over() and len(board.move_stack) < 200:
                if board.turn == chess.WHITE and nebium_is_white or board.turn == chess.BLACK and not nebium_is_white:
                    move = get_nebium_move(model, tokenizer, board, device)
                else:
                    result = engine.play(board, chess.engine.Limit(depth=args.depth))
                    move = result.move
                board.push(move)
                
            res = board.result(claim_draw=True)
            if res == "1-0":
                if nebium_is_white: wins += 1
                else: losses += 1
            elif res == "0-1":
                if not nebium_is_white: wins += 1
                else: losses += 1
            else:
                draws += 1
                
            print(f"Game {i+1}/{args.games} finished. Result: {res}. Score: W:{wins} L:{losses} D:{draws}")
            
    score = (wins + 0.5 * draws) / args.games
    if score == 0:
        score = 0.001
    elif score == 1:
        score = 0.999
        
    import math
    import json
    elo_diff = -400 * math.log10(1/score - 1)
    
    # Assume Stockfish depth 10 is ~2000 Elo
    stockfish_elo = 2000
    estimated_elo = stockfish_elo + elo_diff
    print(f"\nFinal Score: {score*100:.1f}%")
    print(f"Estimated Elo against Stockfish (Depth {args.depth}): {estimated_elo:.0f}")

    if args.output:
        results = {
            "games": args.games,
            "depth": args.depth,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "score": score,
            "estimated_elo": estimated_elo
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
