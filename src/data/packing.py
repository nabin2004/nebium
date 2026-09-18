"""
Sequence packing and batch collation for Nebium.

Implements contiguous document packing of multiple tokenized chess games
into fixed context length tensors with explicit separation semantics ([BOS], [EOS], [SEP]).
"""

import torch


def pack_sequences(
    tokenized_games: list[list[int]],
    max_seq_len: int,
    pad_id: int,
    sep_id: int,
    bos_id: int,
    eos_id: int,
) -> list[dict[str, torch.Tensor]]:
    """
    Packs variable-length tokenized chess games into fixed-length sequence tensors.

    Delineates individual games using [BOS] and [EOS, SEP] special tokens. Padded
    positions are masked in labels with -100 to exclude them from cross-entropy loss.

    Args:
        tokenized_games: List of token ID sequences, each representing one game.
        max_seq_len: Maximum context window length.
        pad_id: Padding token ID ([PAD]).
        sep_id: Separation token ID ([SEP]).
        bos_id: Beginning-of-sequence token ID ([BOS]).
        eos_id: End-of-sequence token ID ([EOS]).

    Returns:
        List of dicts containing 'input_ids', 'labels', and 'attention_mask' tensors.
    """
    examples: list[dict[str, torch.Tensor]] = []
    current_ids: list[int] = []


    for game in tokenized_games:
        game_tokens = [bos_id] + game + [eos_id, sep_id]
        
        if len(current_ids) + len(game_tokens) <= max_seq_len + 1:
            current_ids.extend(game_tokens)
        else:
            # Game doesn't fit, yield current and start new
            if len(current_ids) > 0:
                pad_len = (max_seq_len + 1) - len(current_ids)
                chunk = current_ids + [pad_id] * pad_len
                input_ids = chunk[:-1]
                labels = chunk[1:]
                
                # attention mask: 1 for tokens, 0 for padding
                attention_mask = [1] * (len(current_ids) - 1) + [0] * (pad_len + 1)
                attention_mask = attention_mask[:max_seq_len] # fix size
                labels = [l if m == 1 else -100 for l, m in zip(labels, attention_mask)]
                
                examples.append({
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                })
            
            # Truncate if a single game is longer than max_seq_len
            if len(game_tokens) > max_seq_len + 1:
                game_tokens = game_tokens[:max_seq_len + 1]
            current_ids = game_tokens

    if len(current_ids) > 0:
        pad_len = (max_seq_len + 1) - len(current_ids)
        chunk = current_ids + [pad_id] * pad_len
        input_ids = chunk[:-1]
        labels = chunk[1:]
        attention_mask = [1] * (len(current_ids) - 1) + [0] * (pad_len + 1)
        attention_mask = attention_mask[:max_seq_len]
        labels = [l if m == 1 else -100 for l, m in zip(labels, attention_mask)]
        examples.append({
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        })

    return examples
