import random

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.tokenizer import ChessTokenizer


from src.data.packing import pack_sequences

class MoveSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: list[str],
        tokenizer: ChessTokenizer,
        max_seq_len: int,
    ) -> None:
        self.pad_id = tokenizer.pad_id
        tokenized_games = [tokenizer.encode(seq) for seq in sequences if len(seq.strip()) > 0]
        self.examples = pack_sequences(
            tokenized_games=tokenized_games,
            max_seq_len=max_seq_len,
            pad_id=tokenizer.pad_id,
            sep_id=tokenizer.sep_id,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


def split_sequences(sequences: list[str], train_split: float, seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = list(sequences)
    rng.shuffle(shuffled)
    n_train = max(1, int(len(shuffled) * train_split))
    if n_train >= len(shuffled):
        n_train = max(1, len(shuffled) - 1)
    return shuffled[:n_train], shuffled[n_train:]


def build_dataloaders(
    sequences: list[str],
    tokenizer: ChessTokenizer,
    max_seq_len: int,
    batch_size: int,
    train_split: float,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_seqs, val_seqs = split_sequences(sequences, train_split, seed)
    train_dataset = MoveSequenceDataset(train_seqs, tokenizer, max_seq_len)
    val_dataset = MoveSequenceDataset(val_seqs, tokenizer, max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
