from typing import Any
import chess
import torch


def generate_sample_games(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    prompts: list[str] | None = None,
    max_moves: int = 20,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.95,
) -> list[dict[str, Any]]:
    """
    Generates continuation moves from prompt positions and measures move legality.
    Returns structured results for each sample including legality rate and formatted moves.
    """
    model.eval()
    if prompts is None:
        prompts = ["", "e2e4", "d2d4"]

    samples: list[dict[str, Any]] = []

    with torch.no_grad():
        for prompt in prompts:
            board = chess.Board()
            prompt_str = prompt.strip()
            prompt_valid = True

            # Replay prompt moves onto the board
            if prompt_str:
                for mv in prompt_str.split():
                    try:
                        move = chess.Move.from_uci(mv)
                        if move in board.legal_moves:
                            board.push(move)
                        else:
                            prompt_valid = False
                            break
                    except Exception:
                        prompt_valid = False
                        break

            # Skip generation entirely if the prompt itself is invalid UCI
            if not prompt_valid:
                samples.append({
                    "prompt": prompt_str if prompt_str else "[start]",
                    "generated": "",
                    "full_game": prompt_str,
                    "legal_moves": 0,
                    "total_moves": 0,
                    "legal_rate": 0.0,
                    "stop_reason": "invalid_prompt",
                })
                continue

            # Encode prompt
            if prompt_str:
                encoded = tokenizer.encode(prompt_str)
                input_ids = [tokenizer.bos_id] + encoded
            else:
                input_ids = [tokenizer.bos_id]

            generated_moves: list[str] = []
            legal_count = 0
            stop_reason = "max_moves"

            curr_tokens = list(input_ids)

            for _ in range(max_moves):
                if board.is_game_over():
                    stop_reason = "game_over"
                    break

                max_ctx = getattr(model.module if hasattr(model, "module") else model, "max_seq_len", 512)
                tokens_window = curr_tokens[-max_ctx:]
                x = torch.tensor([tokens_window], dtype=torch.long, device=device)
                mask = torch.ones_like(x, dtype=torch.long, device=device)

                logits = model(x, mask)
                next_token_logits = logits[0, -1, :].clone()

                if temperature > 0.0 and temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                if top_k > 0:
                    val, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    kth = val[-1]
                    next_token_logits[next_token_logits < kth] = -float("Inf")

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    next_token_logits[sorted_indices[sorted_indices_to_remove]] = -float("Inf")

                if temperature == 0.0:
                    next_token = int(next_token_logits.argmax().item())
                else:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    # Fallback if NaNs or all -inf
                    if torch.isnan(probs).any() or probs.sum() <= 0:
                        next_token = int(logits[0, -1, :].argmax().item())
                    else:
                        next_token = int(torch.multinomial(probs, num_samples=1).item())

                # Check for EOS or PAD
                if next_token == getattr(tokenizer, "eos_id", -1) or next_token == getattr(tokenizer, "pad_id", -1):
                    stop_reason = "eos"
                    break

                curr_tokens.append(next_token)
                token_str = tokenizer.decode([next_token]).strip()

                if not token_str:
                    continue

                # Each token might contain one or multiple UCI moves separated by space
                moves_in_token = token_str.split()
                for m_str in moves_in_token:
                    generated_moves.append(m_str)
                    is_legal = False
                    try:
                        move = chess.Move.from_uci(m_str)
                        if move in board.legal_moves:
                            board.push(move)
                            legal_count += 1
                            is_legal = True
                    except Exception:
                        pass
                    # CRITICAL: if the model generated an illegal/unparseable move,
                    # the board state has diverged from the token sequence.
                    # Stop generation here — continuing would evaluate future moves
                    # against the wrong board position, silently inflating legality.
                    if not is_legal:
                        stop_reason = "illegal_move"
                        break
                else:
                    # inner for-loop completed without break → continue outer loop
                    continue
                # inner loop hit 'break' (illegal move) → break outer loop too
                break

            tot = len(generated_moves)
            legal_rate = (legal_count / tot) if tot > 0 else 0.0
            gen_str = " ".join(generated_moves)
            full_str = f"{prompt_str} {gen_str}".strip()

            samples.append({
                "prompt": prompt_str if prompt_str else "[start]",
                "generated": gen_str,
                "full_game": full_str,
                "legal_moves": legal_count,
                "total_moves": tot,
                "legal_rate": legal_rate,
                "stop_reason": stop_reason,
            })

    return samples


def compute_legal_move_rate(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    num_samples: int = 5,
    max_moves: int = 30,
) -> float:
    """
    Computes the percentage of legal moves generated by the model starting from
    an empty board, playing against itself greedily.
    """
    prompts = [""] * num_samples
    samples = generate_sample_games(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompts=prompts,
        max_moves=max_moves,
        temperature=0.0,
    )
    total_legal = sum(s["legal_moves"] for s in samples)
    total_gen = sum(s["total_moves"] for s in samples)
    return (total_legal / total_gen) if total_gen > 0 else 0.0


def evaluate_puzzles(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    puzzles: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Evaluates the model on a dataset of Lichess puzzles.
    Returns overall accuracy and accuracy stratified by rating brackets.
    """
    if not puzzles:
        return {}

    prompts = [p["prompt"] for p in puzzles]
    
    # We only need 1 move to check accuracy. generate_sample_games returns a dict with "generated" string.
    samples = generate_sample_games(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompts=prompts,
        max_moves=1,
        temperature=0.0,
    )
    
    results = {"overall": {"correct": 0, "total": 0}}
    
    for puzzle, sample in zip(puzzles, samples):
        rating = puzzle.get("rating", 1500)
        if rating < 1500:
            bracket = "<1500"
        elif rating < 2000:
            bracket = "1500-2000"
        else:
            bracket = "2000+"
            
        if bracket not in results:
            results[bracket] = {"correct": 0, "total": 0}
            
        solution = puzzle["solution"].strip()
        generated = sample["generated"].strip()
        
        gen_first_move = generated.split()[0] if generated else ""
        is_correct = (gen_first_move == solution)
        
        results["overall"]["total"] += 1
        results[bracket]["total"] += 1
        if is_correct:
            results["overall"]["correct"] += 1
            results[bracket]["correct"] += 1
            
    metrics = {}
    if results["overall"]["total"] > 0:
        metrics["val/puzzle_accuracy"] = results["overall"]["correct"] / results["overall"]["total"]
        
    for bracket in ["<1500", "1500-2000", "2000+"]:
        if bracket in results and results[bracket]["total"] > 0:
            b_name = bracket.replace("<", "under_").replace("+", "_plus").replace("-", "_to_")
            metrics[f"val/puzzle_acc_{b_name}"] = results[bracket]["correct"] / results[bracket]["total"]
            
    return metrics
