import argparse
import time
import chess
import torch
import hydra
from hydra.utils import instantiate
from hydra import compose, initialize
from src.data.prepare import get_tokenizer

def benchmark_generation(model, tokenizer, device, prompt_str, num_moves=5):
    encoded = tokenizer.encode(prompt_str)
    input_ids = [tokenizer.bos_id] + encoded
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(x, dtype=torch.long, device=device)

    def legal_tokens_filter(seq):
        board = chess.Board()
        seq_str = tokenizer.decode(seq).strip()
        moves = seq_str.split()
        for m in moves:
            try:
                board.push_uci(m)
            except:
                return None
        legal_ucis = [m.uci() for m in board.legal_moves]
        allowed_tokens = []
        for v, tok_id in tokenizer.vocab.items():
            s = v.strip()
            if not s:
                continue
            first_move = s.split()[0]
            if first_move in legal_ucis:
                allowed_tokens.append(tok_id)
        if not allowed_tokens:
            return None
        return allowed_tokens

    # Greedy
    start = time.time()
    model.generate(x, mask, max_new_tokens=num_moves, temperature=0.0)
    greedy_time = time.time() - start

    # Top-K
    start = time.time()
    model.generate(x, mask, max_new_tokens=num_moves, temperature=1.0, top_k=40)
    topk_time = time.time() - start

    # Legal filtered Greedy
    start = time.time()
    model.generate(x, mask, max_new_tokens=num_moves, temperature=0.0, legal_tokens_fn=legal_tokens_filter)
    legal_greedy_time = time.time() - start

    # Beam Search
    start = time.time()
    model.beam_search_generate(x, mask, max_new_tokens=num_moves, beam_width=3)
    beam_time = time.time() - start

    print(f"Latency over {num_moves} moves on {device}:")
    print(f"Greedy Decoding: {greedy_time/num_moves:.3f} sec/move")
    print(f"Top-K Sampling: {topk_time/num_moves:.3f} sec/move")
    print(f"Legal-Filtered Greedy: {legal_greedy_time/num_moves:.3f} sec/move")
    print(f"Beam Search (w=3): {beam_time/num_moves:.3f} sec/move")

def main():
    parser = argparse.ArgumentParser(description="Benchmark Latency")
    parser.add_argument("--config_name", type=str, default="nebium_stub")
    args = parser.parse_args()

    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        initialize(version_base=None, config_path="../configs")
    cfg = compose(config_name="config", overrides=[f"model={args.config_name}"])
    
    tokenizer = get_tokenizer(cfg)
    model = instantiate(cfg.model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Warmup
    benchmark_generation(model, tokenizer, device, "e2e4", num_moves=1)
    
    print("\n--- Benchmarking ---")
    benchmark_generation(model, tokenizer, device, "e2e4 e7e5 g1f3", num_moves=5)

if __name__ == "__main__":
    main()
