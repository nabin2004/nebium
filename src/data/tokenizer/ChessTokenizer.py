"""
Byte-Pair Encoding (BPE) Tokenizer for Chess Move Sequences.

Wraps Hugging Face Tokenizers library with whitespace pre-tokenization
treating individual UCI moves as discrete atomic units.
"""

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


class ChessTokenizerHF:
    """
    BPE tokenizer trained specifically on space-delimited UCI chess move sequences.

    Special tokens:
        [PAD], [UNK], [BOS], [EOS], [SEP]
    """

    def __init__(self) -> None:
        """Initializes a BPE model with whitespace pre-tokenization."""
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    def train(self, text_file_path: str, vocab_size: int = 5000) -> None:
        """
        Trains BPE merge rules on a text file containing space-separated UCI move games.

        Args:
            text_file_path: Path to plain text file of move sequences (one game per line).
            vocab_size: Target vocabulary size (default: 5,000).
        """
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"],
            show_progress=True,
        )
        self.tokenizer.train([text_file_path], trainer)

    def encode(self, move_sequence: str) -> list[int]:
        """
        Encodes a space-separated UCI string into a list of token IDs.

        Args:
            move_sequence: Space-separated UCI move string (e.g. 'e2e4 e7e5 g1f3').

        Returns:
            List of integer token IDs.
        """
        return self.tokenizer.encode(move_sequence).ids

    def decode(self, ids: list[int]) -> str:
        """
        Decodes a list of token IDs back into a space-separated UCI move string.

        Args:
            ids: List of integer token IDs.

        Returns:
            Reconstructed string of UCI moves.
        """
        return self.tokenizer.decode(ids)

    def save(self, path: str) -> None:
        """Serializes tokenizer configuration and vocabulary to JSON file."""
        self.tokenizer.save(path)

    def load(self, path: str) -> None:
        """Deserializes tokenizer from a saved JSON file."""
        self.tokenizer = Tokenizer.from_file(path)

    @property
    def vocab_size(self) -> int:
        """Returns total vocabulary size including special tokens."""
        return self.tokenizer.get_vocab_size()

    @property
    def pad_id(self) -> int:
        """Token ID for padding ([PAD])."""
        return self.tokenizer.token_to_id("[PAD]")

    @property
    def bos_id(self) -> int:
        """Token ID for beginning-of-sequence ([BOS])."""
        return self.tokenizer.token_to_id("[BOS]")

    @property
    def eos_id(self) -> int:
        """Token ID for end-of-sequence ([EOS])."""
        return self.tokenizer.token_to_id("[EOS]")

    @property
    def sep_id(self) -> int:
        """Token ID for sequence separator ([SEP])."""
        return self.tokenizer.token_to_id("[SEP]")


ChessTokenizer = ChessTokenizerHF