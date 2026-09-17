import argparse
import os
import chess
import chess.engine
import torch
import hydra
from hydra.utils import instantiate
from hydra import compose, initialize
from src.data.prepare import get_tokenizer
from src.evaluation.chess_metrics import compute_legal_move_rate
from scripts.eval_self_play import ensure_stockfish, get_nebium_move

def main():
    parser = argparse.ArgumentParser(description="Error Analysis & Debugging")
    parser.add_argument("--config_name", type=str, default="nebium_stub")
    parser.add_argument("--checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--games", type=int, default=5, help="Number of games for blunder/loop analysis")
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

    print("1. Computing Illegal Move Rate...")
    legal_rate = compute_legal_move_rate(model, tokenizer, device, num_samples=10, max_moves=20)
    print(f"Legal Move Rate: {legal_rate*100:.1f}%")
    illegal_rate = 1.0 - legal_rate
    print(f"Illegal Move Rate: {illegal_rate*100:.1f}%")
    if illegal_rate > 0.01:
        print("-> Recommendation: Illegal rate > 1%. Enabling move validation layer (legal move masking) is advised.")

    print("\n2. Blunder & Loop Analysis...")
    sf_path = ensure_stockfish()
    
    blunders = 0
    total_moves = 0
    threefold_repetitions = 0
    fifty_move_rules = 0

    with chess.engine.SimpleEngine.popen_uci(sf_path) as engine:
        for i in range(args.games):
            board = chess.Board()
            
            while not board.is_game_over() and len(board.move_stack) < 100:
                # Stockfish eval before move
                info_before = engine.analyse(board, chess.engine.Limit(depth=10))
                score_before = info_before["score"].white().score(mate_score=10000)
                
                # Model plays
                move = get_nebium_move(model, tokenizer, board, device)
                board.push(move)
                total_moves += 1
                
                # Stockfish eval after move
                info_after = engine.analyse(board, chess.engine.Limit(depth=10))
                score_after = info_after["score"].white().score(mate_score=10000)
                
                # If model is white, we want score to stay high. Drop = before - after
                # If model is black, we want score to stay low. Drop = after - before
                is_white = board.turn == chess.BLACK # since we just pushed the move
                if is_white:
                    drop = (score_before - score_after) / 100.0
                else:
                    drop = (score_after - score_before) / 100.0
                    
                if drop > 2.0:
                    blunders += 1
                    
                if board.is_fivefold_repetition() or board.can_claim_threefold_repetition():
                    threefold_repetitions += 1
                    break
                    
                if board.is_seventyfive_moves() or board.can_claim_fifty_moves():
                    fifty_move_rules += 1
                    break
                    
    blunder_rate = (blunders / total_moves) if total_moves else 0.0
    print(f"Total Moves Analyzed: {total_moves}")
    print(f"Blunders (>2.0 pawn drop): {blunders} ({blunder_rate*100:.1f}%)")
    print(f"Games ending in 3-fold repetition: {threefold_repetitions}")
    print(f"Games ending in 50-move rule: {fifty_move_rules}")

    if args.output:
        import json
        results = {
            "legal_move_rate": legal_rate,
            "illegal_move_rate": illegal_rate,
            "total_moves_analyzed": total_moves,
            "blunders": blunders,
            "blunder_rate": blunder_rate,
            "threefold_repetitions": threefold_repetitions,
            "fifty_move_rules": fifty_move_rules
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
