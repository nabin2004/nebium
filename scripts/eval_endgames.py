import argparse
import os
import chess
import chess.engine
import torch
import hydra
from hydra.utils import instantiate
from hydra import compose, initialize
from src.data.prepare import get_tokenizer
from scripts.eval_self_play import ensure_stockfish, get_nebium_move

# Hardcoded sequences of moves that lead to an endgame.
# For simplicity, these are just short sequences where most pieces are traded.
ENDGAMES = {
    "KQ_vs_K_like": "e2e4 d7d5 e4d5 d8d5 d2d4 d5e5 d4e5 f7f5 e5f6 g7f6",
    # This is not a perfect tablebase position, but it simulates a simplified board
}

def main():
    parser = argparse.ArgumentParser(description="Evaluate Endgame Conversions")
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

    sf_path = ensure_stockfish()
    
    results = {}
    with chess.engine.SimpleEngine.popen_uci(sf_path) as engine:
        for name, move_seq in ENDGAMES.items():
            board = chess.Board()
            for m in move_seq.split():
                board.push(chess.Move.from_uci(m))
                
            print(f"\nStarting {name} evaluation from position: {board.fen()}")
            
            nebium_is_white = board.turn == chess.WHITE
            
            # Play 50 moves max
            for _ in range(50):
                if board.is_game_over():
                    break
                    
                if (board.turn == chess.WHITE and nebium_is_white) or (board.turn == chess.BLACK and not nebium_is_white):
                    move = get_nebium_move(model, tokenizer, board, device)
                else:
                    result = engine.play(board, chess.engine.Limit(depth=10))
                    move = result.move
                board.push(move)
                
            res = board.result(claim_draw=True)
            print(f"Result for {name}: {res}")
            results[name] = res

    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
