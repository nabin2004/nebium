import os
import tempfile
import pytest

from src.data.tokenizer import ChessTokenizer


def test_tokenizer_encode_decode():
    tokenizer = ChessTokenizer()
    
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
        f.write("e2e4 e7e5 g1f3\n")
        f.write("d2d4 d7d5 c2c4 c7c6\n")
        temp_path = f.name
    
    try:
        tokenizer.train(temp_path, vocab_size=50)
        
        sequence = "e2e4 e7e5 g1f3"
        encoded = tokenizer.encode(sequence)
        decoded = tokenizer.decode(encoded)
        
        # HuggingFace tokenizers decode might add spaces or preserve them differently,
        # but for space-separated moves, it usually rebuilds the string exactly if we strip it.
        assert sequence.strip() == decoded.strip()

        # Check special tokens
        assert tokenizer.pad_id is not None
        assert tokenizer.bos_id is not None
        assert tokenizer.eos_id is not None
        assert tokenizer.sep_id is not None
    finally:
        os.remove(temp_path)
