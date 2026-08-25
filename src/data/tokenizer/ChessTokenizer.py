from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

class ChessTokenizerHF:
    """
    BPE tokenizer using HuggingFace tokenizers, with moves as pre‑tokens.
    """
    def __init__(self):
        self.tokenizer = Tokenizer(models.BPE())
        # It does not use a standard whitespace pre‑tokenizer because
        # we will handle the move boundaries ourselves.
        # Instead, we'll feed the tokenizer with space‑separated moves,
        # so it sees each move as a whole word.
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    def train(self, text_file_path, vocab_size=5000):
        """
        Train on a file containing one move sequence per line,
        moves space‑separated. Example line: "d2d4 d7d5 c2c4"
        """
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
            show_progress=True
        )
        self.tokenizer.train([text_file_path], trainer)

    def encode(self, move_sequence):
        """
        move_sequence: a string like "d2d4 d7d5 c2c4"
        Returns: list of token IDs
        """
        return self.tokenizer.encode(move_sequence).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def save(self, path):
        self.tokenizer.save(path)

    def load(self, path):
        self.tokenizer = Tokenizer.from_file(path)

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()